weight = None  # path to model weight
resume = False  # whether to resume training process
evaluate = True  # evaluate after each epoch training process
test_only = False  # test process

seed = None  # train process will init a random seed and record
save_path = "exp/default"
num_worker = 32  # total worker in all gpu
batch_size = 16  # total batch size in all gpu
batch_size_val = None  # auto adapt to bs 1 for each gpu
batch_size_test = 1
epoch = 100  # total epoch, data loop = epoch // eval_epoch
eval_epoch = 100  # sche total eval & checkpoint epoch
save_freq = None  # None or int, None indicate only save model last

eval_metric = "mIoU"

sync_bn = False
enable_amp = False
empty_cache = False
find_unused_parameters = False

max_batch_points = 1e8
mix_prob = 0
param_dicts = None  # example: param_dicts = [dict(keyword="block", lr_scale=0.1)]
