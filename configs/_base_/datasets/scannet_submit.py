_base_ = ['scannet.py']

data = dict(
    train=dict(
        split=["train", "val"],
    ),

    val=dict(
        split="val",
    ),

    test=dict(
        split="test",
    ),
)