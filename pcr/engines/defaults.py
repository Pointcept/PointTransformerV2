"""
Default training/testing logic

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import sys
import time
import shutil
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import pointops
from torch.nn.parallel import DistributedDataParallel
from functools import partial
from tensorboardX import SummaryWriter

import pcr.utils.comm as comm
from pcr.datasets import build_dataset, point_collate_fn, collate_fn
from pcr.models import build_model
from pcr.utils.logger import get_root_logger
from pcr.utils.optimizer import build_optimizer
from pcr.utils.scheduler import build_scheduler
from pcr.utils.losses import build_criteria
from pcr.utils.events import EventStorage
from pcr.utils.misc import intersection_and_union_gpu
from pcr.utils.env import get_random_seed, set_seed
from pcr.utils.config import Config, DictAction


def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel model if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression hooks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_local_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_local_rank()]
    ddp = DistributedDataParallel(model,  **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    set_seed(worker_seed)


def default_argument_parser(epilog=None):
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
    Examples:
    Run on single machine:
        $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml
    Change some config options:
        $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001
    Run on multiple machines:
        (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
        (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
    """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config-file', default="", metavar="FILE", help="path to config file")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        # default="tcp://127.0.0.1:{}".format(port),
        default="auto",
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    return parser


def hfai_argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config-file', default="", metavar="FILE", help="path to config file")
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    return parser


def default_config_parser(file_path, options):
    # config name protocol: dataset_name/model_name-exp_name
    if os.path.isfile(file_path):
        cfg = Config.fromfile(file_path)
    else:
        sep = file_path.find("-")
        cfg = Config.fromfile(os.path.join(file_path[:sep], file_path[sep + 1:]))

    if options is not None:
        cfg.merge_from_dict(options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    return cfg


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = cfg.batch_size_val // world_size \
        if cfg.batch_size_val is not None else 1
    # update data loop
    assert cfg.epoch % cfg.eval_epoch == 0
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed * cfg.num_worker_per_gpu + rank
    set_seed(seed)
    return cfg


class Trainer:
    def __init__(self, cfg):
        self.epoch = 0
        self.start_epoch = 0
        self.max_epoch = cfg.eval_epoch
        # TODO: add to hook
        # BestCheckpointer
        self.eval_metric = cfg.eval_metric
        self.best_metric_value = -torch.inf
        # TimeEstimator
        self.iter_end_time = None
        self.max_iter = None

        self.logger = get_root_logger(log_file=os.path.join(cfg.save_path, "train.log"),
                                      file_mode='a' if cfg.resume else 'w')
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.logger.info(f"Save path: {cfg.save_path}")
        self.logger.info(f"Config:\n{cfg.pretty_text}")
        self.storage: EventStorage
        self.logger.info("=> Building model ...")
        self.model = self.build_model()
        self.logger.info("=> Building writer ...")
        self.writer = self.build_writer()
        self.logger.info("=> Building train dataset & dataloader ...")
        self.train_loader = self.build_train_loader()
        self.logger.info("=> Building val dataset & dataloader ...")
        self.val_loader = self.build_val_loader()
        self.logger.info("=> Building criteria, optimize, scheduler, scaler(amp) ...")
        self.criteria = self.build_criteria()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.scaler = self.build_scaler()
        self.logger.info("=> Checking load & resume ...")
        self.resume_or_load()

    def train(self):
        with EventStorage() as self.storage:
            # => before train
            self.logger.info('>>>>>>>>>>>>>>>> Start Training >>>>>>>>>>>>>>>>')
            self.max_iter = self.max_epoch * len(self.train_loader)
            for self.epoch in range(self.start_epoch, self.max_epoch):
                # => before epoch
                if comm.get_world_size() > 1:
                    # fix epoch shuffle pattern make training better
                    self.train_loader.sampler.set_epoch(self.start_epoch)
                self.model.train()
                self.iter_end_time = time.time()
                # => run_epoch
                for i, input_dict in enumerate(self.train_loader):
                    # => before_step
                    # => run_step
                    self.run_step(i, input_dict)
                    # => after_step
                # => after epoch
                self.after_epoch()
            # => after train
            self.logger.info('==>Training done!\nBest {}: {:.4f}'.format(
                self.cfg.eval_metric, self.best_metric_value))
            if self.writer is not None:
                self.writer.close()

    def run_step(self, i, input_dict):
        data_time = time.time() - self.iter_end_time
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output = self.model(input_dict)
            loss = self.criteria(output, input_dict["label"])
        self.optimizer.zero_grad()
        if self.cfg.enable_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()
        if self.cfg.empty_cache:
            torch.cuda.empty_cache()
        n = input_dict["coord"].size(0)
        if comm.get_world_size() > 1:
            loss *= n
            count = input_dict["label"].new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        # intersection, union, target = \
        #     intersection_and_union_gpu(
        #         output.max(1)[1], input_dict["label"], self.cfg.data.num_classes, self.cfg.data.ignore_label)
        # if comm.get_world_size() > 1:
        #     dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
        # intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
        # Here there is no need to sync since sync happened in dist.all_reduce
        batch_time = time.time() - self.iter_end_time
        self.iter_end_time = time.time()

        # self.storage.put_scalar("intersection", intersection)
        # self.storage.put_scalar("union", union)
        # self.storage.put_scalar("target", target)
        self.storage.put_scalar("loss", loss.item(), n=n)
        self.storage.put_scalar("data_time", data_time)
        self.storage.put_scalar("batch_time", batch_time)

        # calculate remain time
        current_iter = self.epoch * len(self.train_loader) + i + 1
        remain_iter = self.max_iter - current_iter
        remain_time = remain_iter * self.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        self.logger.info('Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] '
                         'Scan {batch_size} ({points_num}) '
                         'Data {data_time_val:.3f} ({data_time_avg:.3f}) '
                         'Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) '
                         'Remain {remain_time} '
                         'Lr {lr:.4f} '
                         'Loss {loss:.4f} '.format(epoch=self.epoch + 1, max_epoch=self.max_epoch, iter=i + 1,
                                                   max_iter=len(self.train_loader),
                                                   batch_size=len(input_dict["offset"]),
                                                   points_num=input_dict["offset"][-1],
                                                   data_time_val=data_time,
                                                   data_time_avg=self.storage.history("data_time").avg,
                                                   batch_time_val=batch_time,
                                                   batch_time_avg=self.storage.history("batch_time").avg,
                                                   remain_time=remain_time,
                                                   lr=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                                   loss=loss.item()))
        if i == 0:
            # drop data_time and batch_time for the first iter
            self.storage.history("data_time").reset()
            self.storage.history("batch_time").reset()
        if self.writer is not None:
            self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], current_iter)
            self.writer.add_scalar('train_batch/loss', loss.item(), current_iter)
            # self.writer.add_scalar('train_batch/mIoU', np.mean(intersection / (union + 1e-10)), current_iter)
            # self.writer.add_scalar('train_batch/mAcc', np.mean(intersection / (target + 1e-10)), current_iter)
            # self.writer.add_scalar('train_batch/allAcc', np.sum(intersection) / (np.sum(target) + 1e-10), current_iter)

    def after_epoch(self):
        loss_avg = self.storage.history("loss").avg
        # intersection = self.storage.history("intersection").total
        # union = self.storage.history("union").total
        # target = self.storage.history("target").total
        # iou_class = intersection / (union + 1e-10)
        # acc_class = intersection / (target + 1e-10)
        # m_iou = np.mean(iou_class)
        # m_acc = np.mean(acc_class)
        # all_acc = sum(intersection) / (sum(target) + 1e-10)
        # self.logger.info('Train result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
        #     m_iou, m_acc, all_acc))
        self.logger.info('Train result: loss {:.4f}.'.format(loss_avg))
        current_epoch = self.epoch + 1
        if self.writer is not None:
            self.writer.add_scalar('train/loss', loss_avg, current_epoch)
            # self.writer.add_scalar('train/mIoU', m_iou, current_epoch)
            # self.writer.add_scalar('train/mAcc', m_acc, current_epoch)
            # self.writer.add_scalar('train/allAcc', all_acc, current_epoch)
        self.storage.reset_histories()
        if self.cfg.evaluate:
            self.eval()
        self.save_checkpoint()
        self.storage.reset_histories()

    def eval(self):
        self.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.model.eval()
        self.iter_end_time = time.time()
        for i, input_dict in enumerate(self.val_loader):
            data_time = time.time() - self.iter_end_time
            for key in input_dict.keys():
                input_dict[key] = input_dict[key].cuda(non_blocking=True)
            with torch.no_grad():
                output = self.model(input_dict)
            loss = self.criteria(output, input_dict["label"].long())
            n = input_dict["coord"].size(0)
            if comm.get_world_size() > 1:
                loss *= n
                count = input_dict["label"].new_tensor([n], dtype=torch.long)
                dist.all_reduce(loss), dist.all_reduce(count)
                n = count.item()
                loss /= n
            # TODO: add to evaluator
            pred = output.max(1)[1]
            label = input_dict["label"]
            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(1, input_dict["coord"].float(), input_dict["offset"].int(),
                                            input_dict["origin_coord"].float(), input_dict["origin_offset"].int())

                pred = pred[idx.flatten().long()]
                label = input_dict["origin_label"]
            intersection, union, target = \
                intersection_and_union_gpu(
                    pred, label, self.cfg.data.num_classes, self.cfg.data.ignore_label)
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection), dist.all_reduce(union), dist.all_reduce(target)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            # Here there is no need to sync since sync happened in dist.all_reduce
            batch_time = time.time() - self.iter_end_time
            self.iter_end_time = time.time()

            self.storage.put_scalar("intersection", intersection)
            self.storage.put_scalar("union", union)
            self.storage.put_scalar("target", target)
            self.storage.put_scalar("loss", loss.item(), n=n)
            self.storage.put_scalar("data_time", data_time)
            self.storage.put_scalar("batch_time", batch_time)
            self.logger.info('Test: [{iter}/{max_iter}] '
                             'Data {data_time_val:.3f} ({data_time_avg:.3f}) '
                             'Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) '
                             'Loss {loss:.4f} '.format(iter=i + 1,
                                                       max_iter=len(self.val_loader),
                                                       data_time_val=data_time,
                                                       data_time_avg=self.storage.history("data_time").avg,
                                                       batch_time_val=batch_time,
                                                       batch_time_avg=self.storage.history("batch_time").avg,
                                                       loss=loss.item()))
        loss_avg = self.storage.history("loss").avg
        intersection = self.storage.history("intersection").total
        union = self.storage.history("union").total
        target = self.storage.history("target").total
        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)
        m_iou = np.mean(iou_class)
        m_acc = np.mean(acc_class)
        all_acc = sum(intersection) / (sum(target) + 1e-10)
        self.storage.put_scalar("mIoU", m_iou)
        self.storage.put_scalar("mAcc", m_acc)
        self.storage.put_scalar("allAcc", all_acc)
        self.logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            m_iou, m_acc, all_acc))
        for i in range(self.cfg.data.num_classes):
            self.logger.info('Class_{idx}-{name} Result: iou/accuracy {iou:.4f}/{accuracy:.4f}'.format(
                idx=i, name=self.cfg.data.names[i], iou=iou_class[i], accuracy=acc_class[i]))
        current_epoch = self.epoch + 1
        if self.writer is not None:
            self.writer.add_scalar('val/loss', loss_avg, current_epoch)
            self.writer.add_scalar('val/mIoU', m_iou, current_epoch)
            self.writer.add_scalar('val/mAcc', m_acc, current_epoch)
            self.writer.add_scalar('val/allAcc', all_acc, current_epoch)
        self.logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    def save_checkpoint(self):
        if comm.is_main_process():
            is_best = False
            current_metric_value = self.storage.latest()[self.cfg.eval_metric][0] if self.cfg.evaluate else 0
            if self.cfg.evaluate and current_metric_value > self.best_metric_value:
                self.best_metric_value = current_metric_value
                is_best = True

            filename = os.path.join(self.cfg.save_path, 'model', 'model_last.pth')
            self.logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': self.epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'scaler': self.scaler.state_dict() if self.cfg.enable_amp else None,
                        'best_metric_value': self.best_metric_value},
                       filename+".tmp")
            os.replace(filename+".tmp", filename)
            if is_best:
                shutil.copyfile(filename, os.path.join(self.cfg.save_path, 'model', 'model_best.pth'))
                self.logger.info('Best validation {} updated to: {:.4f}'.format(
                    self.cfg.eval_metric, self.best_metric_value))
            self.logger.info('Currently Best {}: {:.4f}'.format(
                self.cfg.eval_metric, self.best_metric_value))
            if self.cfg.save_freq and self.cfg.save_freq % (self.epoch + 1) == 0:
                shutil.copyfile(filename, os.path.join(self.cfg.save_path, 'model', f'epoch_{self.epoch + 1}.pth'))

    def build_model(self):
        model = build_model(self.cfg.model)
        if self.cfg.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # logger.info(f"Model: \n{self.model}")
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(model.cuda(),
                                 broadcast_buffers=False,
                                 find_unused_parameters=self.cfg.find_unused_parameters)
        return model

    def build_writer(self):
        writer = SummaryWriter(self.cfg.save_path) if comm.is_main_process() else None
        return writer

    def build_train_loader(self):
        train_data = build_dataset(self.cfg.data.train)

        if comm.get_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        else:
            train_sampler = None

        init_fn = partial(
            worker_init_fn, num_workers=self.cfg.num_worker_per_gpu, rank=comm.get_rank(),
            seed=self.cfg.seed) if self.cfg.seed is not None else None

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=self.cfg.batch_size_per_gpu,
                                                   shuffle=(train_sampler is None),
                                                   num_workers=self.cfg.num_worker_per_gpu,
                                                   sampler=train_sampler,
                                                   collate_fn=partial(point_collate_fn,
                                                                      max_batch_points=self.cfg.max_batch_points,
                                                                      mix_prob=self.cfg.mix_prob
                                                                      ),
                                                   pin_memory=True,
                                                   worker_init_fn=init_fn,
                                                   drop_last=True,
                                                   persistent_workers=True)
        return train_loader

    def build_val_loader(self):
        val_loader = None
        if self.cfg.evaluate:
            val_data = build_dataset(self.cfg.data.val)
            if comm.get_world_size() > 1:
                val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
            else:
                val_sampler = None
            val_loader = torch.utils.data.DataLoader(val_data,
                                                     batch_size=self.cfg.batch_size_val_per_gpu,
                                                     shuffle=False,
                                                     num_workers=self.cfg.num_worker_per_gpu,
                                                     pin_memory=True,
                                                     sampler=val_sampler,
                                                     collate_fn=collate_fn)
        return val_loader

    def build_criteria(self):
        return build_criteria(self.cfg.criteria)

    def build_optimizer(self):
        return build_optimizer(self.cfg.optimizer, self.model, self.cfg.param_dicts)

    def build_scheduler(self):
        assert hasattr(self, "optimizer")
        assert hasattr(self, "train_loader")
        self.cfg.scheduler.total_steps = len(self.train_loader) * self.cfg.eval_epoch
        return build_scheduler(self.cfg.scheduler, self.optimizer)

    def build_scaler(self):
        scaler = torch.cuda.amp.GradScaler() if self.cfg.enable_amp else None
        return scaler

    def resume_or_load(self):
        if self.cfg.weight and os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, map_location=lambda storage, loc: storage.cuda())
            load_state_info = self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.cfg.resume:
                self.logger.info(f"Resuming train at eval epoch: {checkpoint['epoch']}")
                self.start_epoch = checkpoint['epoch']
                self.best_metric_value = checkpoint['best_metric_value']
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                if self.cfg.enable_amp:
                    self.scaler.load_state_dict(checkpoint['scaler'])
        else:
            self.logger.info(f"No weight found at: {self.cfg.weight}")