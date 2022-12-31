"""
Render Point Normals Information from raw s3dis data
"""

import os
import argparse
import glob
import tqdm
import multiprocessing as mp
import trimesh
import numpy as np
import open3d


def align_area5b():
    mesh_dir_a = "/home/gofinge/Documents/datasets/Stanford2d3dDataset_noXYZ/area_5a/3d/rgb.obj"
    mesh_dir_b = "/home/gofinge/Documents/datasets/Stanford2d3dDataset_noXYZ/area_5b/3d/rgb.obj"

    mesh_a = open3d.io.read_triangle_mesh(mesh_dir_a)
    mesh_a.triangle_uvs.clear()
    mesh_b = open3d.io.read_triangle_mesh(mesh_dir_b)
    mesh_b.triangle_uvs.clear()
    mesh_b = mesh_b.transform(np.array([[0, 0, -1, -4.09703582],
                                        [0, 1, 0, 0],
                                        [1, 0, 0, -6.22617759],
                                        [0, 0, 0, 1]]))
    os.makedirs("tmp/area_5/3d", exist_ok=True)
    open3d.io.write_triangle_mesh("tmp/area_5/3d/rgb_a.obj", mesh_a)
    open3d.io.write_triangle_mesh("tmp/area_5/3d/rgb_b.obj", mesh_b)
    print("Done")


def parse_object(room_mesh, object_dir, output_dir):
    object_coords = np.loadtxt(object_dir)[:, :3]
    (closest_points, distances, face_id) = room_mesh.nearest.on_surface(object_coords)
    point_normals = room_mesh.face_normals[face_id]
    np.savetxt(os.path.join(output_dir, os.path.basename(object_dir)), point_normals)


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_area', help='number of s3dis data area', default='1')
    parser.add_argument('--raw_data_root', type=str,
                        default='/home/gofinge/Documents/datasets/Stanford2d3dDataset_noXYZ')
    parser.add_argument('--s3dis_data_root', type=str,
                        default='/home/gofinge/Documents/datasets/Stanford3dDataset_v1.2')
    parser.add_argument('--output_root', type=str,
                        default='/home/gofinge/Documents/datasets/Stanford3dDataset_v1.2_normals')
    opt = parser.parse_args()

    # load area mesh
    mesh_dir = os.path.join(opt.raw_data_root, "area_{}".format(opt.data_area), "3d", "rgb.obj")
    mesh = open3d.io.read_triangle_mesh(mesh_dir)
    mesh.triangle_uvs.clear()

    # load room list
    room_name_list = os.listdir(os.path.join(opt.s3dis_data_root, "Area_{}".format(opt.data_area)))
    room_name_list = [room_name for room_name in room_name_list
                      if (room_name != ".DS_Store" and ".txt" not in room_name)]
    bar = tqdm.tqdm(room_name_list)
    pool = mp.Pool(processes=mp.cpu_count())
    for room_name in bar:
        bar.set_postfix_str(room_name)
        room_dir = os.path.join(opt.s3dis_data_root, "Area_{}".format(opt.data_area), room_name)
        output_dir = os.path.join(room_dir.replace(opt.s3dis_data_root, opt.output_root), "Normals")

        # make output dir
        os.makedirs(output_dir, exist_ok=True)
        # get room bound
        room_coords = np.loadtxt(os.path.join(room_dir, "{}.txt".format(room_name)))[:, :3]
        x_min, z_max, y_min = room_coords.min(axis=0)
        x_max, z_min, y_max = room_coords.max(axis=0)
        z_max = - z_max
        z_min = - z_min

        max_bound = np.array([x_max, y_max, z_max]) + 0.1
        min_bound = np.array([x_min, y_min, z_min]) - 0.1
        bbox = open3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        # crop room
        room = mesh.crop(bbox).transform(np.array([[1, 0, 0, 0],
                                                   [0, 0, -1, 0],
                                                   [0, 1, 0, 0],
                                                   [0, 0, 0, 1]]))
        vertices = np.array(room.vertices)
        faces = np.array(room.triangles)
        vertex_normals = np.array(room.vertex_normals)
        room = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals)
        object_dir_list = glob.glob(os.path.join(room_dir, "Annotations", "*.txt"))
        pool.starmap(parse_object, [(room, object_dir, output_dir) for object_dir in object_dir_list])
    pool.close()
    pool.join()


if __name__ == '__main__':
    main_process()
