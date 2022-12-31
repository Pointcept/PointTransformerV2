"""
Unmaintained
it is kept for reference
"""

# dataset settings
dataset_type = "ModelNetDataset"
data_root = "data/modelnet40_normal_resampled"
cache_data = False
names = ["airplane", "bathtub", "bed", "bench", "bookshelf",
         "bottle", "bowl", "car", "chair", "cone",
         "cup", "curtain", "desk", "door", "dresser",
         "flower_pot", "glass_box", "guitar", "keyboard", "lamp",
         "laptop", "mantel", "monitor", "night_stand", "person",
         "piano", "plant", "radio", "range_hood", "sink",
         "sofa", "stairs", "stool", "table", "tent",
         "toilet", "tv_stand", "vase", "wardrobe", "xbox"]

data = dict(
    num_classes=40,
    ignore_label=-1,  # dummy ignore
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=names,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis='x', p=0.5),
            # dict(type="RandomRotate", angle=[-1/24, 1/24], axis='y', p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="Voxelize", voxel_size=0.02, hash_type='fnv', mode='train'),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),

            # dict(type="Voxelize", voxel_size=0.01, hash_type='fnv', mode='train'),
            # dict(type="SphereCrop", point_max=10000, mode='random'),
            # dict(type="CenterShift", apply_z=True),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
        ],
        loop=2,
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="ToTensor"),
        ],
        loop=1,
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="ToTensor"),
        ],
        loop=1,
        test_mode=True,
        test_cfg=dict(
        )
    ),
)

criteria = [
    dict(type="CrossEntropyLoss",
         loss_weight=1.0,
         ignore_index=data["ignore_label"])
]

