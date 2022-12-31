"""
Unmaintained
it is kept for reference
"""

# dataset settings
dataset_type = "ShapeNetPartDataset"
data_root = "data/shapenetcore_partanno_segmentation_benchmark_v0_normal"
cache_data = False
names = ["Airplane_{}".format(i) for i in range(4)] + \
        ["Bag_{}".format(i) for i in range(2)] + \
        ["Cap_{}".format(i) for i in range(2)] + \
        ["Car_{}".format(i) for i in range(4)] + \
        ["Chair_{}".format(i) for i in range(4)] + \
        ["Earphone_{}".format(i) for i in range(3)] + \
        ["Guitar_{}".format(i) for i in range(3)] + \
        ["Knife_{}".format(i) for i in range(2)] + \
        ["Lamp_{}".format(i) for i in range(4)] + \
        ["Laptop_{}".format(i) for i in range(2)] + \
        ["Motorbike_{}".format(i) for i in range(6)] + \
        ["Mug_{}".format(i) for i in range(2)] + \
        ["Pistol_{}".format(i) for i in range(3)] + \
        ["Rocket_{}".format(i) for i in range(3)] + \
        ["Skateboard_{}".format(i) for i in range(3)] + \
        ["Table_{}".format(i) for i in range(3)]

data = dict(
    num_classes=50,
    ignore_label=-1,  # dummy ignore
    names=names,
    train=dict(
        type=dataset_type,
        split=["train", "val"],
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="CenterShift", apply_z=True),
            # dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 24, 1 / 24], axis='x', p=0.5),
            # dict(type="RandomRotate", angle=[-1 / 24, 1 / 24], axis='y', p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomFlip", p=0.5),
            # dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),

            # dict(type="Voxelize", voxel_size=0.01, hash_type='fnv', mode='train'),
            # dict(type="SphereCrop", point_max=2500, mode='random'),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "cls_token", "label"), feat_keys=("coord", "norm"))
        ],
        loop=2,
        test_mode=False,
    ),

    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "cls_token", "label"), feat_keys=("coord", "norm"))
        ],
        loop=1,
        test_mode=False,
    ),

    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="NormalizeCoord"),
            # dict(type="CenterShift", apply_z=True),
        ],
        loop=1,
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "cls_token"), feat_keys=("coord", "norm"))
            ],
            aug_transform=[
                [dict(type="RandomShift2", shift=((0, 0), (0, 0), (0, 0)))],
                [dict(type="RandomShift2", shift=((0.2, 0.2), (0.2, 0.2), (0.2, 0.2)))],
                [dict(type="RandomShift2", shift=((0.2, 0.2), (0.2, 0.2), (-0.2, -0.2)))],
                [dict(type="RandomShift2", shift=((0.2, 0.2), (-0.2, -0.2), (0.2, 0.2)))],
                [dict(type="RandomShift2", shift=((0.2, 0.2), (-0.2, -0.2), (-0.2, -0.2)))],
                [dict(type="RandomShift2", shift=((-0.2, -0.2), (0.2, 0.2), (0.2, 0.2)))],
                [dict(type="RandomShift2", shift=((-0.2, -0.2), (0.2, 0.2), (-0.2, -0.2)))],
                [dict(type="RandomShift2", shift=((-0.2, -0.2), (-0.2, -0.2), (0.2, 0.2)))],
                [dict(type="RandomShift2", shift=((-0.2, -0.2), (-0.2, -0.2), (-0.2, -0.2)))],
            ]
        )
    ),
)

criteria = [
    dict(type="CrossEntropyLoss",
         loss_weight=1.0,
         ignore_index=data["ignore_label"])
]
