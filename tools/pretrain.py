"""
Main Pretaining Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import time
import shutil
import torch
import torch.distributed as dist

import pcr.utils.comm as comm
from pcr.engines.defaults import default_argument_parser, default_config_parser, default_setup, Trainer
from pcr.engines.launch import launch


class PreTrainer(Trainer):
    # Since hook is currently unable, we have to write a trainer for pretraining
    def run_step(self, i, input_dict):
        data_time = time.time() - self.iter_end_time
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
        with torch.cuda.amp.autocast(enabled=self.cfg.enable_amp):
            output = self.model(input_dict)
            loss = output["loss"]
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
        if comm.get_world_size() > 1:
            dist.all_reduce(loss)
            loss = loss / comm.get_world_size()
        # Here there is no need to sync since sync happened in dist.all_reduce
        batch_time = time.time() - self.iter_end_time
        self.iter_end_time = time.time()

        self.storage.put_scalar("loss", loss.item())
        self.storage.put_scalar("data_time", data_time)
        self.storage.put_scalar("batch_time", batch_time)

        # calculate remain time
        current_iter = self.epoch * len(self.train_loader) + i + 1
        remain_iter = self.max_iter - current_iter
        remain_time = remain_iter * self.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        info = ""
        for key in output.keys():
            if key != "loss":
                info += "{name} {value:.3f} ".format(name=key, value=output[key])
        self.logger.info('Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] '
                         'Data {data_time_val:.3f} ({data_time_avg:.3f}) '
                         'Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) '
                         'Remain {remain_time} '
                         'Lr {lr:.4f} '
                         'Loss {loss:.4f} '.format(epoch=self.epoch + 1, max_epoch=self.max_epoch, iter=i + 1,
                                                   max_iter=len(self.train_loader),
                                                   data_time_val=data_time,
                                                   data_time_avg=self.storage.history("data_time").avg,
                                                   batch_time_val=batch_time,
                                                   batch_time_avg=self.storage.history("batch_time").avg,
                                                   remain_time=remain_time,
                                                   lr=self.optimizer.state_dict()['param_groups'][0]['lr'],
                                                   loss=loss.item()) + info
                         )
        if i == 0:
            # drop data_time and batch_time for the first iter
            self.storage.history("data_time").reset()
            self.storage.history("batch_time").reset()
        if self.writer is not None:
            self.writer.add_scalar('lr', self.optimizer.state_dict()['param_groups'][0]['lr'], current_iter)
            self.writer.add_scalar('train_batch/loss', loss.item(), current_iter)

    def after_epoch(self):
        loss_avg = self.storage.history("loss").avg

        self.logger.info('Train result: loss/seg_loss/pos_loss {:.4f}.'.format(
            loss_avg))
        current_epoch = self.epoch + 1
        if self.writer is not None:
            self.writer.add_scalar('train/loss', loss_avg, current_epoch)
        self.storage.reset_histories()
        self.save_checkpoint()

    def save_checkpoint(self):
        if comm.is_main_process():
            filename = os.path.join(self.cfg.save_path, 'model', 'model_last.pth')
            self.logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': self.epoch + 1,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                        'scaler': self.scaler.state_dict() if self.cfg.enable_amp else None,
                        'best_metric_value': self.best_metric_value},
                       filename + ".tmp")
            os.replace(filename + ".tmp", filename)
            if self.cfg.save_freq and self.cfg.save_freq % (self.epoch + 1) == 0:
                shutil.copyfile(filename, os.path.join(self.cfg.save_path, 'model', f'epoch_{self.epoch + 1}.pth'))

    def resume_or_load(self):
        if self.cfg.weight and os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight, map_location=lambda storage, loc: storage.cuda())
            from collections import OrderedDict
            # state_dict = OrderedDict()
            # for key, value in checkpoint['state_dict'].items():
            #     key = key.replace("module.", "")
            #     state_dict[key] = value
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


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = PreTrainer(cfg)
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
