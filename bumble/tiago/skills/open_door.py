import os
import sys
import copy
import numpy as np

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

from bumble.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from bumble.tiago.skills.base import SkillBase, movebase_code2error
import bumble.utils.utils as U
import bumble.utils.vision_utils as VU
import bumble.utils.transform_utils as T # transform_utils
from bumble.tiago.prompters.object_bbox import bbox_prompt_img
import bumble.tiago.RESET_POSES as RP

from termcolor import colored

def make_prompt(query, info, llm_baseline_info=None, method="ours"):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    if method == "ours":
        visual_instructions = [
            "the image",
            "The image is marked with two places: left ('L') and right ('R'). ",
            "",
            "First, describe the scene in image. Describe ",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a description",
            "",
            "either left or right",
            "First, describe ",
        ]
    elif method == "ours_no_markers":
        visual_instructions = [
            "the image",
            "",
            "",
            "First, describe the scene in image. Describe ",
        ]
    else:
        raise NotImplementedError
    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the place where the robot must push the door to open it. You are provided with {visual_instructions[0]} of the scene, and a description of the task. {visual_instructions[1]}You can ONLY select one place to push the door{visual_instructions[2]}. The places approximately indicate suitable point of interaction of the robot with the door, which is likely to be the door handle.

You are a five-time world champion in this game. Output only one of the place: left, or right. Do NOT leave it empty. {visual_instructions[3]}which place is door handle to be more likely at. Then, give a short analysis how you would select the place the robot should interact with on the door. Then, select the place that is easiest to push the door from. Finally, provide the direction in a valid JSON of this format:
{{"place_to_push": ""}}
""".strip()

    if llm_baseline_info:
        instructions += f"""\n
SCENE DESCRIPTION:
{llm_baseline_info['im_scene_desc']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {query}"""
    task_prompt += f"""\n
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

class OpenDoorSkill(SkillBase):
    def __init__(
            self,
            oracle_action=False,
            debug=False,
            use_vlm=False,
            run_dir=None,
            prompt_args=None,
            skip_ros=False,
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.oracle_action = oracle_action
        self.debug = debug
        if not skip_ros:
            self.setup_listeners()
        self.approach_dist = 0.6
        self.pre_goal_dist = 0.8
        self.goal_dist = 0.7 # to the right or left of the door
        self.vis_dir = os.path.join(run_dir, 'open_door')
        os.makedirs(self.vis_dir, exist_ok=True)

        arrow_length_per_pixel = prompt_args.get('arrow_length_per_pixel', 0.15)
        radius_per_pixel = prompt_args.get('radius_per_pixel', 0.06)
        self.prompt_args = {
            "add_arrows": prompt_args.get('add_arrows', True),
            "color": (0, 0, 0),
            "mix_alpha": 0.6,
            'thickness': 2,
            'rgb_scale': 255,
            'plot_dist_factor': prompt_args.get('plot_dist_factor', 1.0),
            'rotate_dist': 0.3,
            'radius_per_pixel': radius_per_pixel,
            'arrow_length_per_pixel': arrow_length_per_pixel,
            'add_object_boundary': False,
            'plot_direction': self.method == "ours",
        }
        self.skill_name = "open_door"
        self.skill_descs = f"""
skill_name: {self.skill_name}
arguments: None
description: Opens the door if the robot is in front of the door. The robot moves forward to open the door.
""".strip()

    def get_param_from_response(self, response, info):
        return_info = {}
        return_info['response'] = response
        direction = None
        error_list = []
        try:
            direction = U.extract_json(response, 'place_to_push')
            print(f"Direction: {direction}")
            if direction.lower() in ['l', 'r']:
                direction = 'left' if direction.lower() == 'l' else 'right'
            if direction.lower() not in ['left', 'right']:
                error = 'Invalid direction. Please provide one of the following locations to push: left, right.'
                error_list.append(error)
                direction = None
        except Exception as e:
            print(f"Error: {e}")
            error = 'Invalid response format. Please provide the direction in a valid JSON format.'
            error_list.append(error)
            direction = None
        return_info['model_out'] = direction
        return_info['poi'] = direction
        return_info['error_list'] = error_list
        return direction, return_info

    def align_with_door(self, env, floor_num, arm='right', execute=True):
        '''
        This function is hard-coded at the moment. It can easily be replaced with a normal prediction.
        '''
        # get the current pose of the robot w.r.t. map
        cur_pos_map, cur_ori_map = self.tf_map.get_transform(target_link='/base_footprint')
        print(colored("Make sure the robot is in front of the door.", 'red'))
        if floor_num == 1:
            ori_map = np.asarray([0.0, 0.0, 0.6943061785962517, 0.719679741526097])
        if floor_num == 2:
            ori_map = np.asarray([0.0, 0.0, 0.7089131175897102, 0.7052958185819889])
        print("sending move_base command")
        _exec = U.confirm_user(True, 'Move to the approach position? (y/n)', 'This will move 50cm behind and align with the door for safety.')
        if not _exec:
            return False
        goal = self.create_move_base_goal((cur_pos_map, ori_map))
        state = self.send_move_base_goal(goal)

        rospy.sleep(1.0)
        tar_pos_base = np.asarray([-0.5, 0.0, 0.0])
        transform = T.pose2mat((self.tf_odom.get_transform(target_link=f'/base_footprint')))
        target_pos_odom = transform @ np.concatenate((tar_pos_base, [1.0]))
        target_pos_odom = target_pos_odom[:3]
        self.goto_odom_pos(env, target_pos_odom, speed=0.5, threshold=0.02)
        return

    def step(self, env, rgb, depth, pcd, normals, query, arm='right', execute=True, run_vlm=True, info=None, **kwargs):
        '''
            action: Position, Quaternion (xyzw) of the goal
        '''
        if env is not None:
            # get the current position of the robot_base w.r.t. odom
            self.send_head_command(head_positions=[0.0, -0.4])
            obs_pp = VU.get_obs(env, self.tf_base)
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

        text_direction = ''
        # because we do not have any parameters at the moment for this skill, we will just use a predefined motion and initialize some variables to avoid errors
        prompt_rgb = rgb.copy()
        response = ''
        return_info = {
            'response': response,
            'model_out': '',
            'error_list': [],
        }

        door_distance = None
        if self.oracle_action:
            # ask the user for direction
            # poi is w.r.t to the base_footprint frame
            clicked_points = U.get_user_input(rgb)
            poi = pcd[clicked_points[0][1], clicked_points[0][0]]
            # get the arm_left_5_link position w.r.t. base_footprint
            tool_pos, tool_ori = self.tf_base.get_transform(target_link=f'/arm_{arm}_5_link')
            # get the door distance with only x distance in base_footprint frame
            door_distance = np.abs(poi[0] - tool_pos[0])
        if run_vlm:
            img_size = min(rgb.shape[0], rgb.shape[1])
            prompt_rgb = rgb.copy()
            info_cp = copy.deepcopy(info)
            info_cp['bbox_ignore_ids'] = []
            info_cp['bbox_id2dist'] = {1: 0.0, 2: 0.0}
            if self.prompt_args['plot_direction']:
                # prompt the image with left / right directions to predict.
                prompt_args = copy.deepcopy(self.prompt_args)
                prompt_args.update({
                    'radius': int(img_size * self.prompt_args['radius_per_pixel']),
                    'fontsize': int(img_size * 30 * self.prompt_args['radius_per_pixel']),
                    'start_point': (rgb.shape[1]//2, rgb.shape[0]//2),
                    'label_list': ['L', 'R'],
                })

                # two bboxes are used to show the left and right directions.
                center_pt_l = (int(img_size * 0.2), rgb.shape[0]//2)
                center_pt_r = (int(rgb.shape[1] - img_size * 0.2), rgb.shape[0]//2)
                bboxes_dir = [(1, center_pt_l[0], center_pt_l[1], center_pt_l[0], center_pt_l[1]), \
                    (2, center_pt_r[0], center_pt_r[1], center_pt_r[0], center_pt_r[1])]
                prompt_rgb, _  = bbox_prompt_img(prompt_rgb, bboxes_dir, prompt_args, info_cp)

            step_idx = info['step_idx']

            if self.method == "ours_no_markers":
                prompt_rgb = rgb.copy()

            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_{info["save_key"]}.png'))
            encoded_image = U.encode_image(prompt_rgb)
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=None,
                make_prompt_func=make_prompt,
                make_prompt_func_kwargs={
                    'query': query,
                    'info': info_cp,
                },
            )
            text_direction, return_info = self.get_param_from_response(response, info)
            self.save_model_output(
                rgb=prompt_rgb,
                response=response,
                subtitles=[f'Task Query: {query}', f''],
                img_file=os.path.join(self.vis_dir, f'output_{info["save_key"]}.png'),
            )

            # get the bboxes for the door
            bboxes, final_mask_image = self.get_object_bboxes(rgb, query=['door'])
            if len(bboxes) == 0:
                return self.on_failure(
                    reason_for_failure="Door handle not found in the image.",
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )
            # used mainly for debugging
            overlay_image = U.overlay_xmem_mask_on_image(
                rgb.copy(),
                np.array(final_mask_image),
                use_white_bg=False,
                rgb_alpha=0.3
            )
            U.save_image(overlay_image, os.path.join(self.vis_dir, f'overlay_image_{info["save_key"]}.png'))
            unqiue_labels = np.unique(final_mask_image)
            # find the one with highest area
            door_mask = final_mask_image != 0
            pcd_door = pcd[door_mask]
            # remove nan values
            pcd_door = pcd_door[~np.isnan(pcd_door).any(axis=1)]
            # take the minimum x value of the door_pcd
            x_min = np.min(pcd_door[:, 0])
            print(f"Door distance: {x_min}")
            door_distance = x_min
        capture_history = {
            'image': prompt_rgb,
            'query': query,
            'model_response': text_direction,
            'full_response': response,
            'text_direction': text_direction,
            'model_analysis': '',
        }
        self.save_model_output(
            rgb=prompt_rgb,
            response=response,
            subtitles=[f'Task Query: {query}', f''],
            img_file=os.path.join(self.vis_dir, f'output_{info["save_key"]}.png'),
        )

        error = None
        if len(return_info['error_list']) > 0:
            error = "Following errors have been produced: "
            for e in return_info['error_list']:
                error += f"{e}, "
            error = error[:-2]
            return self.on_failure(
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        if env is None:
            return self.on_failure(
                reason_for_failure='Environment is None.', # only for debugging
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )
        overrite_ori_map = np.asarray([0.0, 0.0, 0.7089131175897102, 0.7052958185819889])
        # We use move_base to approach the door. Maintain the 30cm distance from the door
        approach_pos_base = np.asarray([door_distance - self.approach_dist, 0.0, 0.0])
        approach_ori_base = np.asarray([0.0, 0.0, 0.0, 1.0])
        transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
        approach_pose_map = T.pose2mat((approach_pos_base, approach_ori_base))
        approach_pose_map = transform @ approach_pose_map
        approach_pos_map = approach_pose_map[:3, 3]
        approach_ori_map = T.mat2quat(approach_pose_map[:3, :3])

        # Use goto_odom_pos to move 20 cm ahead.
        goto_pose_odom = T.pose2mat((np.asarray([door_distance + self.pre_goal_dist, 0.0, 0.0]), approach_ori_map))
        transform = T.pose2mat((self.tf_odom.get_transform(target_link=f'/base_footprint')))
        goto_pose_odom = transform @ goto_pose_odom
        goto_pos_odom = goto_pose_odom[:3, 3]
        goto_ori_odom = T.mat2quat(goto_pose_odom[:3, :3])

        # use move_base to move further 1m ahead
        goal_pos_base = np.asarray([door_distance + self.pre_goal_dist + self.goal_dist, 0.0, 0.0]) # if text_direction == 'left' else np.asarray([door_distance + self.pre_goal_dist, -self.goal_dist, 0.0])
        goal_ori_base = np.asarray([0.0, 0.0, 0.0, 1.0])
        # if left, then the robot should rotate 90 degrees to the left otherwise 90 degrees to the right
        if text_direction == 'left':
            goal_ori_base = np.asarray([0.0, 0.0, 0.7071068, 0.7071068])
        else:
            goal_ori_base = np.asarray([0.0, 0.0, -0.7071068, 0.7071068])
        transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
        goal_pose_map = T.pose2mat((goal_pos_base, goal_ori_base))
        goal_pose_map = transform @ goal_pose_map
        goal_pos_map = goal_pose_map[:3, 3]
        goal_ori_map = T.mat2quat(goal_pose_map[:3, :3])

        if self.debug:
            pcd_wrt_map = np.concatenate((pcd.reshape(-1, 3), np.ones((pcd.reshape(-1,3).shape[0], 1))), axis=1)
            transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
            pcd_wrt_map = (transform @ pcd_wrt_map.T).T
            pcd_wrt_map = pcd_wrt_map[:, :3]
            pcd_to_plot = pcd_wrt_map.reshape(-1,3)
            rgb_to_plot = rgb.reshape(-1,3)

            # current pos in grey
            cur_pos_map, _ = self.tf_map.get_transform(target_link='/base_footprint')
            cur_pos_map = np.asarray(cur_pos_map)
            pcd_to_plot = np.concatenate((pcd_to_plot, cur_pos_map.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[128.0, 128.0, 128.0]])), axis=0) # current pos in grey

            # approach pos in red
            pcd_to_plot = np.concatenate((pcd_to_plot, approach_pos_map.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0) # approach pos in green

            # goto pos in green
            transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/odom')))
            goto_pos_map = np.concatenate((goto_pos_odom, [1.0]))
            goto_pos_map = transform @ goto_pos_map
            goto_pos_map = goto_pos_map[:3]
            pcd_to_plot = np.concatenate((pcd_to_plot, goto_pos_map.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[0.0, 255.0, 0.0]])), axis=0) # goto pos in green

            # goal pos in blue
            pcd_to_plot = np.concatenate((pcd_to_plot, goal_pos_map.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[0.0, 0.0, 255.0]])), axis=0)

            U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)

        is_success = False
        error = None
        print(colored(f"Door distance: {door_distance}", 'red'))
        print(colored(f"Make sure the door handle is to the {text_direction} of the robot.", 'red'))
        print(colored(f"Make sure there is enough space in the {text_direction} side of the robot so that the arm can move.", "red"))
        if execute:
            user_input = input("Press Enter to continue or 'q' to quit: ")
            if user_input == 'q':
                execute = False
        if execute:
            if text_direction == 'left':
                _exec = U.reset_env(env, reset_pose=RP.HOME_R_OPEN_DOOR_L, int_pose=RP.INT_L_H, delay_scale_factor=1.5)
            else:
                _exec = U.reset_env(env, reset_pose=RP.HOME_L_OPEN_DOOR_R, int_pose=RP.INT_R_H, delay_scale_factor=1.5)
            print("Moving to the approach position")
            goal = self.create_move_base_goal((approach_pos_map, overrite_ori_map))
            state = self.send_move_base_goal(goal)

            # ask the user if to continue opening the door
            print("Moving to the goto position?")
            _continue = input("Press Enter to continue or 'q' to quit: ")
            if _continue == 'q':
                return self.on_failure(
                    reason_for_failure="Did not execute the skill.",
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )

            # reset the torso to the open door position because move_base will go back to a lower torso position.
            env.reset(reset_arms=False, reset_pose=RP.OPEN_DOOR_R if arm == 'right' else RP.OPEN_DOOR_L, allowed_delay_scale=6.0)
            # bang-bang control to move to the goto position with 5cm threshold
            self.goto_odom_pos(env, goto_pos_odom, speed=0.5, threshold=0.05)
            print("Reached the goto position")
            # wait for 5 seconds
            rospy.sleep(1.0)

            # move to the goal position. This will go very far ahead of the door
            print("Moving to the goal position")

            goal = self.create_move_base_goal((goal_pos_map, goal_ori_map))
            state = self.send_move_base_goal(goal)
            is_success = True
            if text_direction == 'left':
                U.reset_env(env, reset_pose=RP.INT_L_H, delay_scale_factor=1.5) # leave the arm in the intermediate position
            else:
                U.reset_env(env, reset_pose=RP.INT_R_H, delay_scale_factor=1.5)

        if not is_success:
            return self.on_failure(
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        return self.on_success(
            capture_history=capture_history,
            return_info=return_info,
        )
