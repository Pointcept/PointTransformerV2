# Point Transformer V2
<p align="center">
  <img src="figures/design.png" width="480">
</p>

This is a lightweight and easy-to-use codebase for point cloud recognition research supporting indoor & outdoor point cloud datasets and several backbones, namely **PointCloudRecog** (PCR). The next release version of PCR will further support instance segmentation, object detection, and pretraining.
This is an official implementation of the following paper:

- **Point Transformer V2: Grouped Vector Attention and Partition-based Pooling**   
*Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao*  
Neural Information Processing Systems (NeurIPS) 2022  
[ [Arxiv](https://arxiv.org/abs/2210.05666) ] [ [Bib](https://xywu.me/research/ptv2/bib.txt) ]

## News
- *Dec, 2022*: Initial release our PCR codebase and PTv2 official implementation.
- *Sep, 2022*: [PTv2](https://arxiv.org/abs/2210.05666) accepted by NeurIPS 2022.

## Overview

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

### Requirements
- Ubuntu: 18.04 or higher
- CUDA: 10.2 or higher
- PyTorch: 1.10.0 ~ 1.11.0
- Hardware: 4 x 24G memory GPUs or better

### Conda Environment

```bash
conda create -n pcr python=3.8 -y
conda activate pcr
conda install ninja -y
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx yapf addict einops scipy plyfile termcolor timm -y
conda install -c pyg pytorch-cluster pytorch-scatter pytorch-sparse -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113
```

### Optional Installation

```bash
# Open3D (Visualization)
pip install open3d

# PTv1 & PTv2
cd libs/pointops
python setup.py install
cd ../..

# stratified transformer
pip install torch-points3d

# fix dependence, caused by install torch-points3d 
pip uninstall SharedArray
pip install SharedArray==3.2.1

cd libs/pointops2
python setup.py install
cd ../..

# MinkowskiEngine (SparseUNet)
# refer https://github.com/NVIDIA/MinkowskiEngine

# torchsparse (SPVCNN)
# refer https://github.com/mit-han-lab/torchsparse
# install method without sudo apt install
conda install google-sparsehash -c bioconda
export C_INCLUDE_PATH=${CONDA_PREFIX}/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:CPLUS_INCLUDE_PATH
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```

## Data Preparation

### ScanNet v2

The preprocessing support semantic and instance segmentation for both `ScanNet20` and `ScanNet200`.

- Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
- Run preprocessing code for raw ScanNet as follows:

```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset (output dir).
python pcr/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```

- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset.
mkdir data
ln -s ${RAW_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

### S3DIS

- Download S3DIS data by filling this [Google form](https://docs.google.com/forms/d/e/1FAIpQLScDimvNMCGhy_rmBA2gHfDu3naktRm6A8BPwAWWDv-Uhm6Shw/viewform?c=0&w=1). Download the `Stanford3dDataset_v1.2_Aligned_Version.zip` file and unzip it.
- The original S3DIS data contains some bugs data need manually fix it. `xxx^@xxx`
- Run preprocessing code for S3DIS as follows:

```bash
# RAW_S3DIS_DIR: the directory of downloaded Stanford3dDataset_v1.2_Aligned_Version dataset.
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset (output dir).
python pcr/datasets/preprocessing/s3dis/preprocess_s3dis.py --dataset_root ${RAW_S3DIS_DIR} --output_root ${PROCESSED_S3DIS_DIR}
```
- Link processed dataset to codebase.
```bash
# PROCESSED_S3DIS_DIR: the directory of processed S3DIS dataset.
mkdir data
ln -s ${RAW_S3DIS_DIR} ${CODEBASE_DIR}/data/s3dis
```

### Semantic KITTI
- Download [Semantic KITTI](http://www.semantic-kitti.org/dataset.html#download) dataset.
- Link dataset to codebase.
```bash
# SEMANTIC_KITTI_DIR: the directory of Semantic KITTI dataset.
mkdir data
ln -s ${SEMANTIC_KITTI_DIR} ${CODEBASE_DIR}/data/semantic_kitti
```

## Quick Start

### Training
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d scannet -c semseg-ptv2m2-0-base -n semseg-ptv2m2-0-base
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/scannet/semseg-ptv2m2-0-base.py --options save_path=exp/scannet/semseg-ptv2m2-0-base
```
**Resume training from checkpoint.** If the training process is interrupted by accident, the following script can resume training from a given checkpoint.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
# simply add "-r true"
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME} -r true
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH} resume=True weight=${CHECKPOINT_PATH}
```

### Testing
The validation during training only evaluate model on point clouds after grid sampling (voxelization) and testing is need to achieve a precise evaluate result.
Our testing code support TTA (test time augmentation) testing.
(Currently only support testing on a single GPU, I might add support to multi-gpus testing in the future version.)

```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```

For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d scannet -n semseg-ptv2m2-0-base -w model_best
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/scannet/semseg-ptv2m2-0-base.py --options save_path=exp/scannet/semseg-ptv2m2-0-base weight=exp/scannet/semseg-ptv2m2-0-base/models/model_best.pth
```

### Offset
`Offset` is the separator of point clouds in batch data, and it is similar to the concept of `Batch` in PyG. 
A visual illustration of batch and offset is as follows:
<p align="center">
  <img src="figures/offset.png" width="480">
</p>

## Model Zoo

The PCR codebase supports most combinations of supporting datasets and models, and I haven't tested and tuned all the combinations. Consequently, I list configs for some of them. It would be helpful to reach me if you find a better setting.

Currently, PCR only focuses on semantic segmentation, but in the next version, I will introduce the hook mechanism to support pretraining, instance segmentation, and object detection. The released version contains some code for classification, but I did not maintain it since object-level recognition is quite different from scene-level recognition. I will focus more on the scene-level point clouds. Nevertheless, the code framework support object classification. If you want to run classification tasks, you can modify the codebase from these unmaintained codes for object classification.

**Recommendations:** 
- PTv2m2 (good performance)
- SparseUNet-SpConv (fast, lightweight and easy to install)

(Please email me your recommendations, I might add support in a future version)

### Point Transformer V1 & V2
- **PTv2 mode2 (recommend)**

The original PTv2 was trained on 4 * RTX a6000 (48G memory). Even enabling AMP, the memory cost of the original PTv2 is slightly larger than 24G. Considering GPUs with 24G memory are much more accessible, I tuned the PTv2 on the latest PCR codebase and made it runnable on 4 * RTX 3090 machines.

`PTv2 Mode2` enables AMP and disables _Position Encoding Multiplier_ & _Grouped Linear_. During our further research, we found that precise coordinates are not necessary for point cloud understanding (Replacing precise coordinates with grid coordinates doesn't influence the performance. Also, SparseUNet is an example). As for Grouped Linear, my implementation of Grouped Linear seems to cost more memory than the Linear layer provided by PyTorch. Benefiting from the codebase and better parameter tuning, we also relieve the overfitting problem. The reproducing performance is even better than the results reported in our paper.

Example running script is as follows:

```bash
# ptv2m2: PTv2 mode2, disable PEM & Grouped Linear, GPU memory cost < 24G (recommend)
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-ptv2m2-0-base -n semseg-ptv2m2-0-base
# ScanNet test benchmark (train on train set and val set)
sh scripts/train.sh -g 4 -d scannet -c semseg-ptv2m2-1-benchmark-submit -n semseg-ptv2m2-1-benchmark-submit
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-ptv2m2-0-base -n semseg-ptv2m2-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-ptv2m2-0-base -n semseg-ptv2m2-0-base
```

Example training and testing records are as follows:

|     Dataset     | mIoU | mAcc | allAcc | Config |                            Train                             |                             Test                             |                         Tensorboard                          |
| :-------------: | :--: | :--: | :----: | :----: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  ScanNet v2 20  | 75.5 | 82.9 |  91.2  | [config](configs/scannet/semseg-ptv2m2-0-base.py) | [log](https://xywu.me/research/pcr/logs/semseg-scannet20-ptv2m2/train.log) | [log](https://xywu.me/research/pcr/logs/semseg-scannet20-ptv2m2/test.log) | [tensorboard](https://tensorboard.dev/experiment/2eyHRBfrRyyywUqej9yh2w/#scalars&_smoothingWeight=0) |
|  ScanNet v2 200 | 31.9 | 39.2 |  82.7  | [config](configs/scannet200/semseg-ptv2m2-0-base.py) | [log](https://xywu.me/research/pcr/logs/semseg-scannet200-ptv2m2/train.log) | [log](https://xywu.me/research/pcr/logs/semseg-scannet200-ptv2m2/test.log) | [tensorboard](https://tensorboard.dev/experiment/V62nQuVgQKeaBAJKP59juw/#scalars&_smoothingWeight=0) |
|  S3DIS Area 5   | 72.6 | 78.0 |  91.6  | [config](configs/s3dis/semseg-ptv2m2-0-base.py) | [log](https://xywu.me/research/pcr/logs/semseg-s3dis-ptv2m2/train.log) | [log](https://xywu.me/research/pcr/logs/semseg-s3dis-ptv2m2/test.log) | [tensorboard](https://tensorboard.dev/experiment/DKbqkNvGTX6BxdHZx6Vi7Q/#scalars&_smoothingWeight=0) |

`*Dataset` represents reported results from an older version of the PCR codebase.

- **PTv2 mode1**

`PTv2 mode1` is the original PTv2 we reported in our paper, example running script is as follows:

```bash
# ptv2m1: PTv2 mode1, Original PTv2, GPU memory cost > 24G
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-ptv2m1-0-base -n semseg-ptv2m1-0-base
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-ptv2m1-0-base -n semseg-ptv2m1-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-ptv2m1-0-base -n semseg-ptv2m1-0-base
```

- **PTv1**

The original PTv1 is also available in our PCR codebase. I haven't run PTv1 for a long time, but I have ensured that the example running script works well. 

```bash
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-ptv1-0-base -n semseg-ptv1-0-base
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-ptv1-0-base -n semseg-ptv1-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-ptv1-0-base -n semseg-ptv1-0-base
```


### Stratified Transformer
1. Uncomment `# from .stratified_transformer import *` in `pcr/models/__init__.py`.
2. Refer [Optional Installation](installation) to install dependence.
3. Training with the following example running scripts:
```bash
# stv1m1: Stratified Transformer mode1, Modified from the original Stratified Transformer code.
# ptv2m2: Stratified Transformer mode2, My rewrite version (recommend).

# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-stv1m2-0-refined -n semseg-stv1m2-0-refined
sh scripts/train.sh -g 4 -d scannet -c semseg-stv1m1-0-origin -n semseg-stv1m1-0-origin
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-stv1m2-0-refined -n semseg-stv1m2-0-refined
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-stv1m2-0-refined -n semseg-stv1m2-0-refined
```
*I did not tune the parameters for Stratified Transformer and just ensured it could run.*

### SparseUNet

The PCR codebase provides `SparseUNet` implemented by `SpConv` and `MinkowskiEngine`. The SpConv version is recommended since SpConv is easy to install and faster than MinkowskiEngine. Meanwhile, the PCR codebase will add more support to instance segmentation and object detection in the future version. SpConv is also widely applied in these areas.

- **SpConv (recommend)**

The SpConv version `SparseUNet` in the codebase was fully rewrite from [Li Jiang](https://llijiang.github.io/)'s code, example running script is as follows:

```bash
# Uncomment "# from .sparse_unet import *" in "pcr/models/__init__.py"
# Uncomment "# from .spconv_unet import *" in "pcr/models/sparse_unet/__init__.py"
# ScanNet val
sh scripts/train.sh -g 4 -d scannet -c semseg-spunet34c-0-base -n semseg-spunet34c-0-base
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-spunet34c-0-base -n semseg-spunet34c-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-spunet34c-0-base -n semseg-spunet34c-0-base
# Semantic-KITTI
sh scripts/train.sh -g 2 -d semantic-kitti -c semseg-spunet34c-0-base -n semseg-spunet34c-0-base
```

Example training and testing records are as follows:

|    Dataset     | mIoU | mAcc | allAcc | Config |                            Train                             | Test |                         Tensorboard                          |
| :------------: | :--: | :--: | :----: | :----: | :----------------------------------------------------------: | :--: | :----------------------------------------------------------: |
| ScanNet v2 20  | 73.6 | 82.2 |  90.4  | [config](configs/scannet/semseg-spunet34c-0-base.py) | [log](https://xywu.me/research/pcr/logs/semseg-scannet20-spunet34c/train.log) | log  | [tensorboard](https://tensorboard.dev/experiment/A9w7yGCjRe6l4gwzNX2kOA/#scalars&_smoothingWeight=0) |
| ScanNet v2 200 | 28.8 | 36.1 |  81.1  | [config](configs/scannet200/semseg-spunet34c-0-base.py) | [log](https://xywu.me/research/pcr/logs/semseg-scannet200-spunet34c/train.log) | [log](https://xywu.me/research/pcr/logs/semseg-scannet200-spunet34c/test.log)  | [tensorboard](https://tensorboard.dev/experiment/oJWSEHBeTeOmT7M5uva5Ig/#scalars&_smoothingWeight=0) |
|  S3DIS Area 5  | 67.7 | 73.1 |  90.1  | [config](configs/s3dis/semseg-spunet34c-0-base.py) | [log](https://xywu.me/research/pcr/logs/semseg-s3dis-spunet34c/train.log) | [log](https://xywu.me/research/pcr/logs/semseg-s3dis-spunet34c/test.log) | [tensorboard](https://tensorboard.dev/experiment/XGfX1SBBTpq0MrfQ6gO4IQ/#scalars&_smoothingWeight=0) |


`*Dataset ` represents reported results from an older version of the PCR codebase.

- **MinkowskiEngine**

The MinkowskiEngine version `SparseUNet` in the codebase was modified from original MinkowskiEngine repo, and example running script is as follows:

```bash
# Uncomment "# from .sparse_unet import *" in "pcr/models/__init__.py"
# Uncomment "# from .mink_unet import *" in "pcr/models/sparse_unet/__init__.py"
# ScanNet
sh scripts/train.sh -g 4 -d scannet -c semseg-minkunet34c-0-base -n semseg-minkunet34c-0-base
# ScanNet200
sh scripts/train.sh -g 4 -d scannet200 -c semseg-minkunet34c-0-base -n semseg-minkunet34c-0-base
# S3DIS
sh scripts/train.sh -g 4 -d s3dis -c semseg-minkunet34c-0-base -n semseg-minkunet34c-0-base
# Semantic-KITTI
sh scripts/train.sh -g 2 -d semantic-kitti -c semseg-minkunet34c-0-base -n semseg-minkunet34c-0-base
```

### SPVCNN
`SPVCNN` is baseline model of [SPVNAS](https://github.com/mit-han-lab/spvnas), it is also a practical baseline for outdoor dataset.

```bash
# Semantic-KITTI
sh scripts/train.sh -g 2 -d semantic-kitti -c semseg-spvcnn34c-0-base -n semseg-spvcnn34c-0-base
```

## Citation
If you find this work useful to your research, please cite our work:
```
@inproceedings{wu2022point,
  title     = {Point transformer V2: Grouped Vector Attention and Partition-based Pooling},
  author    = {Wu, Xiaoyang and Lao, Yixing and Jiang, Li and Liu, Xihui and Zhao, Hengshuang},
  booktitle = {NeurIPS},
  year      = {2022}
}
```

## Acknowledgement
The repo is derived from Point Transformer code and inspirited by several repos, e.g., [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [pointnet2](https://github.com/charlesq34/pointnet2), [mmcv](https://github.com/open-mmlab/mmcv/tree/master/mmcv), and [Detectron2](https://github.com/facebookresearch/detectron2).
