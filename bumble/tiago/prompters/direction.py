import os
import sys
import copy
import numpy as np

import cv2
from scipy.spatial.transform import Rotation as R

import bumble.utils.utils as U
import bumble.utils.transform_utils as T
import bumble.utils.vision_utils as VU

def get_prompt_to_add(
        im,
        direction,
        prompt_args,
        info,
    ):
    '''
    Args:
        im: np.array, image
        direction: str, direction to add the prompt
        prompt_args: dict, arguments for the prompt
        info: dict, information about the current state

        info dict: plot_robot_pos, plot_robot_ori, skill_type, move_dist, rotate_angle, cam_intr, cam_extr
        Example prompt_args:
            prompt_args = {
                "add_arrows": add_arrows,
                "arrow_length": arrow_length,
                "color": (0, 0, 0),
                "start_point": (im.shape[1]//2, im.shape[0] - 20) if _type == 'move' else (im.shape[1]//2, int(im.shape[0] - 1)),
                "mix_alpha": 0.6,
                'radius': int(img_size * radius_per_pixel),
                'thickness': 2,
                'fontsize': int(img_size * 30 * radius_per_pixel),
                'rgb_scale': 255,
                'plot_dist_factor': plot_dist_factor,
                'rotate_dist': 0.3
            }
    '''
    arrow_length = prompt_args['arrow_length']
    start_point = prompt_args.get('start_point', (im.shape[1]//2, im.shape[0] - 2*arrow_length))
    color = prompt_args.get('color', (0, 0, 0))
    robot_pos = copy.deepcopy(info['plot_robot_pos'])
    robot_ori = copy.deepcopy(info['plot_robot_ori'])
    assert ('skill_type' in info.keys()) or ('skill_type' in prompt_args.keys()), "skill_type is not provided."
    skill_type = info['skill_type'] if 'skill_type' in info.keys() else prompt_args['skill_type']
    dist = None
    if skill_type == 'move':
        dist = info['move_dist'] if 'move_dist' in info.keys() else prompt_args['move_dist']
    if skill_type == 'rotate':
        dist = info['rotate_angle'] if 'rotate_angle' in info.keys() else prompt_args['rotate_angle']
    end_point = None
    text_to_add = None
    if skill_type == 'move':
        direction_wrt_robot = None
        if direction == 'forward':
            direction_wrt_robot = np.array([4, 0, 0]) # forward is not visible if we use 1
            text_to_add = 'F'
        elif direction == 'backward':
            direction_wrt_robot = np.array([-1, 0, 0])
            text_to_add = 'B'
        elif direction == 'left':
            direction_wrt_robot = np.array([0, 1.0, 0])
            text_to_add = 'L'
        elif direction == 'right':
            direction_wrt_robot = np.array([0, -1.0, 0])
            text_to_add = 'R'
        pose_wrt_robot = dist * direction_wrt_robot * prompt_args['plot_dist_factor']
        # add the pose wrt robot to the robot's pose
        new_robot_pos = robot_pos + pose_wrt_robot
        # transform = T.pose2mat((info['plot_robot_pos'], info['plot_robot_ori']))
        # new_robot_pos = (transform @ np.array([pose_wrt_robot[0], pose_wrt_robot[1], 0, 1]))[:3]
    elif skill_type == 'rotate':
        rotate_ori = R.from_quat(robot_ori)
        if direction == 'left':
            rotate_mat = R.from_euler('z', dist, degrees=True)
            text_to_add = 'L'
        elif direction == 'right':
            rotate_mat = R.from_euler('z', -dist, degrees=True)
            text_to_add = 'R'
        else:
            raise NotImplementedError("Rotate skill is not implemented yet.")
        move_ori = rotate_mat * rotate_ori
        pose_wrt_robot = prompt_args['rotate_dist']*np.array([1, 0, 0])
        transform = T.pose2mat((info['plot_robot_pos'], move_ori.as_quat()))
        new_robot_pos = (transform @ np.array([pose_wrt_robot[0], pose_wrt_robot[1], 0, 1]))[:3]

    end_point = VU.pos2pixels(new_robot_pos.reshape(1,-1), info['cam_intr'], info['cam_extr'])[0]
    # check if end_point is nan
    if np.isnan(end_point).any():
        # keep the end point as top of the image, i.e., y = 0 and x = robot_pos[0]
        end_point = (int(robot_pos[0]), 0)

    end_point = (int(end_point[0]), int(end_point[1]))
    # cap the end point to be within the image with at least prompt_args['radius'] distance from the border
    end_point = (
        max(prompt_args['radius'], min(im.shape[1] - prompt_args['radius'], end_point[0])),
        max(prompt_args['radius'], min(im.shape[0] - prompt_args['radius'], end_point[1])),
    )
    return (start_point, end_point, text_to_add)

def add_arrows_to_image(
        im,
        vis_prompts,
        prompt_args,
        info,
        skip_arrow=False,
    ):
    overlay = im.copy() # we do mixing with the original image
    white = (
        prompt_args['rgb_scale'],
        prompt_args['rgb_scale'],
        prompt_args['rgb_scale'],
    )
    text_radius = prompt_args['radius'] / 3**0.5
    char_len_adjust = 1.0
    center_adjust = 1.0
    color = prompt_args['color']
    radius = prompt_args['radius']
    for prompt in vis_prompts:
        start_point, end_point, text = prompt
        if not skip_arrow:
            # draw the arrow
            cv2.arrowedLine(
                overlay,
                start_point,
                end_point,
                color, prompt_args['thickness'],
            )
        cv2.circle(
            overlay,
            end_point,
            radius, color, thickness=-1,
        )
        cv2.circle(overlay, end_point, radius, white, -1)
        fontsize = prompt_args['fontsize']/im.shape[0]
        cv2.putText(
            overlay,
            text,
            (int(end_point[0]-(text_radius/center_adjust)),int(end_point[1]+text_radius)),
            cv2.FONT_HERSHEY_SIMPLEX,
            char_len_adjust*fontsize, color, prompt_args['thickness'],
            cv2.LINE_AA,
        )
    im = cv2.addWeighted(
        overlay, prompt_args['mix_alpha'],
        im, 1 - prompt_args['mix_alpha'],
        0,
    )
    return im

def add_circle_to_image(
        im,
        vis_prompts,
        prompt_args,
        info,
    ):
    overlay = im.copy() # we do mixing with the original image
    white = (
        prompt_args['rgb_scale'],
        prompt_args['rgb_scale'],
        prompt_args['rgb_scale'],
    )
    text_radius = prompt_args['radius'] / 3**0.5
    char_len_adjust = 1.0
    center_adjust = 1.0
    color = prompt_args['color']
    radius = prompt_args['radius']
    for prompt in vis_prompts:
        end_point, text = prompt
        cv2.circle(
            overlay,
            end_point,
            radius, color, thickness=-1,
        )
        cv2.circle(overlay, end_point, radius, white, -1)
        fontsize = prompt_args['fontsize']/im.shape[0]
        cv2.putText(
            overlay,
            text,
            (int(end_point[0]-(text_radius/center_adjust)),int(end_point[1]+text_radius)),
            cv2.FONT_HERSHEY_SIMPLEX,
            char_len_adjust*fontsize, color, prompt_args['thickness'],
            cv2.LINE_AA,
        )
    im = cv2.addWeighted(
        overlay, prompt_args['mix_alpha'],
        im, 1 - prompt_args['mix_alpha'],
        0,
    )
    return im


def prompt_move_img(
        im,
        prompt_args,
        info,
    ):
    vis_prompts = []
    prompt_to_add = []
    f_p = get_prompt_to_add(im, 'forward', prompt_args, info)
    b_p = get_prompt_to_add(im, 'backward', prompt_args, info)
    l_p = get_prompt_to_add(im, 'left', prompt_args, info)
    r_p = get_prompt_to_add(im, 'right', prompt_args, info)
    prompt_to_add = [f_p, l_p, r_p]
    im = add_arrows_to_image(im, prompt_to_add, prompt_args, info)
    return im

def prompt_rotate_img(
        im,
        prompt_args,
        info,
    ):
    vis_prompts = []
    prompt_to_add = []
    l_p = get_prompt_to_add(im, 'left', prompt_args, info)
    r_p = get_prompt_to_add(im, 'right', prompt_args, info)
    prompt_to_add = [l_p, r_p]
    im = add_arrows_to_image(im, prompt_to_add, prompt_args, info)
    return im
