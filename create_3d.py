import numpy as np
import skvideo.io
import sys
import quaternion
from tqdm import tqdm
import open3d as o3d


path = sys.argv[1]
video = skvideo.io.vread(path)
video = video.astype(np.float32) / 255.
H, W = video.shape[1], video.shape[2]

data = np.load(path[:-3] + 'npz')
depths, pos, rot = data['depth'], data['pos'], data['rot']
depths = depths[..., 0]

hfov = 90 * np.pi / 180
K = np.array([
    [1 / np.tan(hfov / 2.), 0., 0., 0.],
    [0., 1 / np.tan(hfov / 2.), 0., 0.],
    [0., 0.,  1, 0],
    [0., 0., 0, 1]
])

points, colors = [], []

for t in tqdm(list(range(video.shape[0]))):
    rgb = video[t]
    depth = depths[[t]]
    xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, W))
    xs = xs.reshape(1, W, W)
    ys = ys.reshape(1, W, W)

    xys = np.vstack((xs * depth, ys * depth, -depth, np.ones(depth.shape)))
    xys = xys.reshape(4, -1)
    xy_c0 = np.matmul(np.linalg.inv(K), xys)

    quaternion_0 = quaternion.from_float_array(rot[t])
    translation_0 = pos[t]
    rotation_0 = quaternion.as_rotation_matrix(quaternion_0)
    T_world_camera0 = np.eye(4)
    T_world_camera0[:3, :3] = rotation_0
    T_world_camera0[:3, 3] = translation_0

    xy_c1 = T_world_camera0 @ xy_c0
    xy_c1 = xy_c1[:-1]

    points.append(xy_c1.T)
    colors.append(rgb.reshape(-1, 3))
points = np.concatenate(points)
colors = np.concatenate(colors)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.visualization.draw_geometries([pcd])