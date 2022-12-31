"""
Preprocessing Script for S3DIS

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import argparse
import glob
import multiprocessing as mp
import numpy as np
import torch


def parse_room(room, source_root, save_root, parse_normals=False):
    if isinstance(room, list):
        room, angle = room
    else:
        angle = None

    print("Parsing: {}".format(room))
    classes = ["ceiling", "floor", "wall", "beam", "column", "window",
               "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]
    class2label = {cls: i for i, cls in enumerate(classes)}
    source_dir = os.path.join(source_root, room)
    save_path = os.path.join(save_root, room) + ".pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    object_path_list = sorted(glob.glob(os.path.join(source_dir, 'Annotations/*.txt')))

    room_coords = []
    room_normals = []
    room_colors = []
    room_semantic_gt = []
    room_instance_gt = []

    for object_id, object_path in enumerate(object_path_list):
        object_name = os.path.basename(object_path).split("_")[0]
        obj = np.loadtxt(object_path)
        coords = obj[:, :3]
        colors = obj[:, 3: 6]
        # note: in some room there is 'stairs' class
        class_name = object_name if object_name in classes else "clutter"
        semantic_gt = np.repeat(class2label[class_name], coords.shape[0])
        semantic_gt = semantic_gt.reshape([-1, 1])
        instance_gt = np.repeat(object_id, coords.shape[0])
        instance_gt = instance_gt.reshape([-1, 1])

        room_coords.append(coords)
        room_colors.append(colors)
        room_semantic_gt.append(semantic_gt)
        room_instance_gt.append(instance_gt)

        if parse_normals:
            object_norm_dir = os.path.join(source_dir, "normals", object_name)
            normals = np.loadtxt(object_norm_dir)
            assert normals.shape[0] == coords.shape[0]
            room_normals.append(normals)

    room_coords = np.ascontiguousarray(np.vstack(room_coords))
    room_coords -= room_coords.mean(0)
    rot_t = None
    if angle is not None:
        angle = (2 - angle / 360) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        room_coords = room_coords @ np.transpose(rot_t)
    room_colors = np.ascontiguousarray(np.vstack(room_colors))
    room_semantic_gt = np.ascontiguousarray(np.vstack(room_semantic_gt))
    room_instance_gt = np.ascontiguousarray(np.vstack(room_instance_gt))
    save_dict = dict(coord=room_coords, color=room_colors, semantic_gt=room_semantic_gt, instance_gt=room_instance_gt)
    if parse_normals:
        room_normals = np.ascontiguousarray(np.vstack(room_normals))
        if rot_t is not None:
            room_normals = room_normals @ np.transpose(rot_t)
        save_dict["normal"] = room_normals
    torch.save(save_dict, save_path)


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', type=str,
                        default='/home/gofinge/Documents/datasets/Stanford3dDataset_v1.2_Aligned_Version')
                        # default='/home/gofinge/Documents/datasets/Stanford3dDataset_v1.2')
    parser.add_argument('--output_root', type=str,
                        default='/home/gofinge/Documents/datasets/processed/s3dis')
                        # default='/home/gofinge/Documents/datasets/processed/s3dis_aligned_with_normals')
    parser.add_argument('--parse_normals', action='store_true')
    opt = parser.parse_args()
    room_list = []

    for i in range(1, 7):
        if "Aligned_Version" in opt.source_root:
            area_dir = os.path.join(opt.source_root, "Area_{}".format(i))
            room_name_list = os.listdir(area_dir)
            room_name_list = [room_name for room_name in room_name_list
                              if (room_name != ".DS_Store" and ".txt" not in room_name)]
            room_list += [os.path.join("Area_{}".format(i), room_name) for room_name in room_name_list]
        else:
            area_dir = os.path.join(opt.source_root, "Area_{}".format(i))
            align_dir = os.path.join(area_dir, "Area_{}_alignmentAngle.txt".format(i))
            room_name_list = np.loadtxt(align_dir, dtype=str)
            room_list += [[os.path.join("Area_{}".format(i), room_name[0]), int(room_name[1])]
                          for room_name in room_name_list]

    pool = mp.Pool(processes=mp.cpu_count())
    pool.starmap(parse_room, [(room, opt.source_root, opt.save_root, opt.parse_normals)
                              for room in room_list])
    pool.close()
    pool.join()


if __name__ == '__main__':
    main_process()