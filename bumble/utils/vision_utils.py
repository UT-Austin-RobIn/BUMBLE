import os
import time
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import bumble.utils.transform_utils as T # transform_utils

def look_at_rotate_z(start_position, tar_position):
    '''
        start_position: np.ndarray (2,)
        tar_position: np.ndarray (2,)
    '''
    # Calculate direction vector
    direction = tar_position - start_position
    direction_norm = np.linalg.norm(direction)

    # Normalize direction
    direction /= direction_norm

    # Calculate angle to rotate around Z-axis
    angle = np.arctan2(direction[1], direction[0])

    # Construct quaternion representing the rotation around Z-axis
    rotation = R.from_euler('z', angle, degrees=False)
    return rotation.as_quat()

def pcd_from_depth(
        depth,
        intrinsic_matrix,
        depth_trunc=5,
        depth_scale=1,
    ):

    width, height = depth.shape[:2]
    pinholecameraIntrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height,
        intrinsic_matrix=intrinsic_matrix
    )

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d.geometry.Image(depth),
        pinholecameraIntrinsic,
        depth_trunc=depth_trunc,
        depth_scale=depth_scale,
        project_valid_depth_only=False,
    )
    return pcd

def estimate_normals(pts, cam_pos, radius=0.1, max_nn=30):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.reshape(-1,3))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
    pcd.orient_normals_towards_camera_location(cam_pos)
    normals = np.asarray(pcd.normals)
    return normals

def pixels2pos(
        pixels,
        depth,
        cam_intr,
        cam_extr=None,
        return_normal=False,
    ):
    '''
        pixels: np.ndarray(N, 2) or (2,)
        depth: np.ndarray(H,W)
        cam_intr: 3x3
        cam_extr: 4x4 if not None
        Return:
            pos: (N, 3)
    '''
    assert isinstance(pixels, np.ndarray), "Pixels should be of np.ndarray"
    single_dim = False
    if pixels.ndim == 1:
        single_dim = True
        pixels = pixels[None, :]

    pcd = pcd_from_depth(
        depth=depth,
        intrinsic_matrix=cam_intr,
    )

    pts = np.asarray(pcd.points) / 1000.0 # unit conversion
    pts = np.concatenate((pts, np.ones((pts.shape[0],1))), axis=-1)
    pts = cam_extr @ pts.T
    pts = pts[:3,:].T

    pts = pts.reshape(depth.shape[0], depth.shape[1], 3)
    pos = np.stack([pts[pt[1], pt[0]] for pt in pixels], axis=0)

    if single_dim:
        pos = pos.reshape(-1)

    if return_normal:
        # find index of nan
        pts_p = pts.reshape(-1,3)
        mask = ~np.isnan(pts_p).any(axis=1)
        pts_p = pts_p[mask]
        normal_calc = estimate_normals(
            pts_p.reshape(-1,3),
            cam_pos=cam_extr[:3,3],
            radius=0.1,
            max_nn=30,
        )
        normals = np.zeros_like(pts.reshape(-1,3))
        normals[mask] = normal_calc
        normals = normals.reshape(depth.shape[0], depth.shape[1], 3)
        return pos, pts, normals

    return pos, pts, None

def pos2pixels(
        pos,
        cam_intr,
        cam_extr,
    ):
    '''
        pos: np.ndarray(N, 3)
        cam_intr: 3x3
        cam_extr: 4x4
        Return:
            pixels: np.ndarray(N, 2)
    '''
    pts = np.concatenate((pos, np.ones((pos.shape[0],1))), axis=-1)
    pts = np.linalg.inv(cam_extr) @ pts.T
    pts = pts[:3,:].T
    pts = pts @ cam_intr.T
    pts = pts / pts[:,2][:,None]
    return pts[:,:2]

def get_obs(env, tf_listener, return_normal=False):
    obs = env._observation()
    rgb = obs['tiago_head_image'][:, :, ::-1].astype(np.uint8) # BGR -> RGB
    depth = obs['tiago_head_depth']
    cam_intr = np.asarray(list(env.cameras['tiago_head'].camera_info.K)).reshape(3,3)
    cam_pose = tf_listener.get_transform('/xtion_optical_frame')
    cam_extr = T.pose2mat(cam_pose)
    pos, pcd, normals = pixels2pos(
        np.asarray([(rgb.shape[0]//2, rgb.shape[1]//2)]),
        depth=depth.astype(np.float32),
        cam_intr=cam_intr,
        cam_extr=cam_extr,
        return_normal=return_normal,
    )
    return {
        'rgb': rgb,
        'depth': depth,
        'cam_intr': cam_intr,
        'cam_extr': cam_extr,
        'pcd': pcd,
        'normals': normals,
    }
