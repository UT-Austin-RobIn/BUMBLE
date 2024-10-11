#!/usr/bin/env python3
import os
import cv2
import sys
import copy
import numpy as np
import pickle
from math import pi
from termcolor import colored
from scipy.spatial.transform import Rotation as R

### ROS IMPORTS
import rospy
import actionlib
import moveit_commander
import moveit_msgs.msg
from control_msgs.msg import JointTrajectoryControllerState
import geometry_msgs.msg
from std_msgs.msg import String
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

import bumble
from bumble.tiago.prompters.wrappers import GroundedSamWrapper
from bumble.tiago.skills.base import SkillBase, movebase_code2error
from bumble.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
import bumble.utils.utils as U
import bumble.utils.transform_utils as T # transform_utils
import bumble.utils.vision_utils as VU # vision_utils
from bumble.tiago.prompters.object_bbox import bbox_prompt_img

import moveit_commander
moveit_commander.roscpp_initialize(sys.argv)

def clearout_mask_image(mask_image, unique_values):
    final_mask_image = np.zeros_like(mask_image)
    for val in unique_values:
        mask = mask_image == val
        # if there multiple connected components, keep the largest one
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        largest_label = np.argmax(stats[1:, 4]) + 1
        mask = labels == largest_label
        final_mask_image += mask * val
    return final_mask_image

def get_button_positions(image, mask_image):
    # find all the circles in the mask_image
    unique_values = np.unique(mask_image)
    unique_values = unique_values[unique_values != 0]
    mask_image = clearout_mask_image(mask_image, unique_values)
    segms = {}

    selected_values = sorted(unique_values, key=lambda x: np.sum(mask_image == x), reverse=True)[:4] # 4 is a safe number

    final_mask_image = np.zeros_like(mask_image)
    for val in selected_values:
        final_mask_image += ((mask_image == val)*val).astype(np.uint8)

    unique_values = np.unique(final_mask_image)
    bboxes = []
    for label in unique_values:
        mask = final_mask_image == label
        # get the bbox of the mask
        nonzero = np.nonzero(mask)
        xmin, ymin, xmax, ymax = np.min(nonzero[1]), np.min(nonzero[0]), np.max(nonzero[1]), np.max(nonzero[0])
        bboxes.append([label, xmin, ymin, xmax, ymax])

    # sort the bbox by lowest y value first
    bboxes = sorted(bboxes, key=lambda x: x[2])
    return bboxes, final_mask_image

def make_prompt(query, info, llm_baseline_info=None, method="ours"):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    if method == "ours":
        visual_instructions = [
            "the image of the scene marked",
            "image",
            "The button_id is the character marked in circle next to the button, example, 'B'. ",
            "marked ",
            " id",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a description of the scene",
            "button description section",
            "",
            "",
            " id",
        ]
    elif method == "ours_no_markers":
        visual_instructions = [
            "the image of the scene",
            "image",
            "",
            "",
            "",
        ]
    else:
        raise NotImplementedError
    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the button{visual_instructions[4]} that the robot must push to complete the task. You are provided with {visual_instructions[0]} with button{visual_instructions[4]}s, and the task description. You can ONLY select the button{visual_instructions[4]} present in the {visual_instructions[1]}. You are currently in floor number {info['floor_num']}.

You are a five-time world champion in this game. Output only one button{visual_instructions[4]}, do NOT leave it empty. {visual_instructions[2]}Avoid using numericals for the button_id. First, summarize all the errors made in previous predictions if provided. Then, describe the task you want to achieve, specify whether you want to go floor higher or lower from the current floor. Then, describe all the {visual_instructions[3]}buttons in the scene along with the corresponding functions using common sense. List down all the marked ids that should not be pressed, they could fire key insertion place, fire buttons. Then, give a short analysis of how you would chose the button. Then, select button that must be pressed to complete the task. Finally, provide the button{visual_instructions[4]} that must be pressed in a valid JSON of this format:
{{"button_id": ""}}
"""

    if llm_baseline_info:
        instructions += f"""\n
SCENE DESCRIPTION:
{llm_baseline_info['im_scene_desc']}
BUTTON ID DESCRIPTIONS:
{llm_baseline_info['obj_descs']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {query}. You are currently in floor number {info['floor_num']}."""
    task_prompt += f"""
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

def make_history_call_elevator(history):
    instructions = f"""
Below is some of the prediction history that can help you understand the mistakes and successful predictions made in the past. Pay close attention to the mistakes made in the past and try to avoid them in the current prediction. Summarize each of prediction failures in the past. Based on the history, improve your prediction. Note that the button ids are marked differently in each image.
PREDICTION HISTORY:
"""
    history_desc = []
    history_model_analysis = []
    for ind, msg in enumerate(history):
        if ('model_analysis' not in msg) or (msg['model_analysis'] == '') or (msg['model_analysis'] is None):
            msg['model_analysis'] = 'The prediction was accurate for successful task completion.'
        example_desc = f"""\n
Example {ind+1}:
- Task Query: {msg['query']}
- Answer: {msg['model_response']}
""".strip()
        history_desc.append(example_desc)
        history_model_analysis.append(msg['model_analysis'])
    return instructions, history_desc, history_model_analysis

def make_history_use_elevator(history):
    instructions = f"""
Below is some of the prediction history that can help you understand the mistakes and successful predictions made in the past. Pay close attention to the mistakes made in the past and try to avoid them in the current prediction. Summarize each of prediction failures in the past. Based on the history, improve your prediction. Note that the button ids are marked differently in each image.
PREDICTION HISTORY:
"""
    history_desc = []
    history_model_analysis = []
    for ind, msg in enumerate(history):
        if ('model_analysis' not in msg) or (msg['model_analysis'] == '') or (msg['model_analysis'] is None):
            msg['model_analysis'] = 'The prediction was accurate for successful task completion.'
        example_desc = f"""\n
Example {ind+1}:
- Task Query: {msg['query']}
- Answer: {{"button_id": {msg['model_response'][0]}, "target_floor_num": {msg['model_response'][1]}}}
""".strip()
        history_desc.append(example_desc)
        history_model_analysis.append(msg['model_analysis'])
    return instructions, history_desc, history_model_analysis

def make_prompt_floor_ch(query, info, llm_baseline_info=None, method="ours"):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    if method == "ours":
        visual_instructions = [
            "the image of the scene marked",
            "image",
            "The button_id is the character marked in circle next to the button, example, 'B'. ",
            "marked ",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a description of the scene",
            "button description section",
            "",
            "",
        ]
    else:
        raise NotImplementedError
    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the button id that the robot must push to complete the task and the floor the robot will reach after pressing the button. You are provided with {visual_instructions[0]} with button ids, and the task description. You can ONLY select the button id present in the {visual_instructions[1]}. You are currently in floor number {info['floor_num']}.

You are a five-time world champion in this game. Output only one button id and floor num, do NOT leave it empty. {visual_instructions[2]}Avoid using numericals for the button_id. First, describe the task you want to achieve, specify whether you want to go up/down. Then, describe all the {visual_instructions[3]}buttons in the scene along with the corresponding functions using common sense. List down all the marked ids that should not be pressed, they could fire key insertion place, fire buttons. Then, give a short analysis of how you would chose the button. Then, select button that must be pressed to complete the task. Finally, provide the button id that must be pressed along with the floor the robot will reach after pressing the button in a valid JSON of this format:
{{"button_id": "", "target_floor_num": ""}}
"""

    if llm_baseline_info:
        instructions += f"""\n
SCENE DESCRIPTION:
{llm_baseline_info['im_scene_desc']}
BUTTON ID DESCRIPTIONS:
{llm_baseline_info['obj_descs']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {query}"""
    task_prompt += f"""
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

class ElevatorSkill(SkillBase):
    def __init__(
            self,
            oracle_position: bool = False,
            use_vlm: bool = True,
            adjust_gripper: bool = True,
            debug: bool = False,
            run_dir: str = None,
            prompt_args: dict = None,
            skip_ros: bool = False,
            add_histories: bool = False,
            *args, **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.oracle_position = oracle_position
        if not skip_ros:
            self.setup_listeners()
        self.adjust_gripper = adjust_gripper
        self.adjust_gripper_length = 0.17 # 85
        self.debug = debug
        self.add_histories = add_histories
        radius_per_pixel = prompt_args.get('radius_per_pixel', 0.03)
        self.prompt_args = {
            "color": (0, 0, 0),
            "mix_alpha": 0.6,
            'thickness': 2,
            'rgb_scale': 255,
            'add_object_boundary': prompt_args.pop('add_object_boundary', False),
            'add_dist_info': prompt_args.pop('add_dist_info', False),
            'add_arrows_for_path': prompt_args.pop('add_arrows_for_path', False),
            'radius_per_pixel': radius_per_pixel,
            'plot_outside_bbox': True,
        }
        if self.add_histories:
            history_eval_dirs = self.get_history_dirs()
            history_list = []
            for hist_eval_dir in history_eval_dirs:
                samples_per_hist = 1
                _history_all_path = os.path.join(hist_eval_dir, 'history_all.pkl')
                if hist_eval_dir.endswith('.pkl'):
                    _history_all_path = hist_eval_dir
                assert os.path.exists(_history_all_path), f"History file not found: {_history_all_path}"
                _history_list = pickle.load(open(_history_all_path, 'rb'))
                if not isinstance(_history_list, list):
                    _history_list = [_history_list]
                # _success_list = [h for h in _history_list if h['is_success']]
                _history_list = [h for h in _history_list if not h['is_success']]
                _history_list = _history_list[:samples_per_hist]
                # _success_list = _success_list[:samples_per_hist]
                history_list.extend(_history_list)
            self.history_list = history_list
            print(f"Loaded {len(history_list)} failed samples.")

    def get_history_dirs(self):
        raise NotImplementedError

    def get_approach_pose(self, pos, normal, arm, frame):
        '''
            pos, normal: np.ndarray are w.r.t. the base_footprint
        '''
        assert frame in ['odom', 'base_footprint']
        approach_pos = pos + np.asarray([-0.1, 0.0, 0.0])
        approach_ori = R.from_rotvec(np.asarray([3.0*np.pi/2, 0.0, 0.0])).as_quat()
        if frame == 'odom':
            transform = T.pose2mat(self.tf_odom.get_transform('/base_footprint'))
            approach_pose = T.pose2mat((approach_pos, approach_ori))
            approach_pose = transform @ approach_pose
            approach_pos, approach_ori = T.mat2pose(approach_pose)
        return approach_pos, approach_ori

    def get_goto_pose(self, pos, normal, arm, approach_ori, frame):
        goto_pos = pos - np.asarray([0.0, 0.0, 0.0])
        if self.close_gripper_var:
            # add 1cm to the z-axis
            goto_pos[2] += 0.01
        goto_ori = approach_ori
        return goto_pos, goto_ori

    def get_object_bboxes(self, rgb, query, gsam=None):
        # query is a list of object names
        if (self._gsam is None) and (gsam is None):
            self._gsam = GroundedSamWrapper(sam_ckpt_path=os.environ['SAM_CKPT_PATH'])
        if gsam is None:
            gsam = self._gsam
        bboxes = []
        final_mask_image = gsam.segment(rgb, query)
        final_mask_image = np.array(final_mask_image)
        bboxes, final_mask_image = get_button_positions(rgb, final_mask_image)
        return bboxes, final_mask_image

    def get_param_from_response(self, response, obj_bbox_list, pcd, mask_image):
        '''
            skill_specific function to get the param from the vlm response
        '''
        return_info = {}
        return_info['response'] = response
        return_info['error_list'] = []
        object_id = ''
        try:
            object_id = U.extract_json(response, 'button_id')
            print(f"Buton ID: {object_id}")
        except Exception as e:
            print(str(e))
            object_id = ''
        return_info['button_id'] = object_id
        bbox_selected = [bbox for bbox in obj_bbox_list if bbox.obj_id.lower() == object_id.lower()]
        if len(bbox_selected) == 0:
            error = f"Object id {object_id} not found in the scene."
            return_info['error_list'].append(error)
            return None, object_id, return_info

        bbox_env_id = bbox_selected[0].env_id
        bbox = bbox_selected[0].bbox
        mask = mask_image == bbox_env_id
        return_info['bbox'] = bbox
        return_info['object_id_mask'] = mask
        coord = (int((bbox[1]+bbox[3])/2.0), int((bbox[2]+bbox[4])/2.0))
        return_info['coord'] = coord
        return coord, object_id, return_info

    def get_param_from_response_floor_num(self, response, return_info):
        '''
            skill_specific function to get the param from the vlm response
        '''
        floor_num = ''
        try:
            floor_num = U.extract_json(response, 'target_floor_num')
            print(f"Floor number: {floor_num}")
        except Exception as e:
            print(str(e))
            floor_num = ''
        return_info['floor_num'] = floor_num
        return_info['target_floor_num'] = floor_num
        return floor_num, return_info

    def press_once(self, env, rgb, depth, pcd, normals, arm, query, execute, run_vlm, info, make_prompt_func, make_history_prompt, adjust_y=False, bld=None, history=None, close_gripper=False, **kwargs):
        assert bld is not None
        self.close_gripper_var = close_gripper
        if execute and close_gripper:
            self.close_gripper(env=env, side='right')
        if run_vlm:
            if self.add_histories:
                if history is None:
                    history = []
                history.extend(self.history_list)
            gsam_query = ["buttons"]
            bboxes, mask_image = self.get_object_bboxes(rgb, query=gsam_query)
            if len(bboxes) == 0:
                error = "No elevator buttons in the scene."
                return self.on_failure(
                    reason_for_failure=error,
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )
            overlay_image = U.overlay_xmem_mask_on_image(
                rgb.copy(),
                np.array(mask_image),
                use_white_bg=False,
                rgb_alpha=0.3
            )
            step_idx = info['step_idx']
            # save the overlay image for debugging
            U.save_image(overlay_image, os.path.join(self.vis_dir, f'overlay_image_{info["save_key"]}.png'))
            img_size = min(mask_image.shape)
            self.prompt_args['radius'] = int(img_size * self.prompt_args['radius_per_pixel'])
            self.prompt_args['fontsize'] = int(img_size * 30 * self.prompt_args['radius_per_pixel'])

            bbox_id2dist = {}
            for bbox in bboxes:
                center = (bbox[1] + bbox[3]) // 2, (bbox[2] + bbox[4]) // 2
                pos_wrt_base = pcd[center[1], center[0]]
                dist = np.linalg.norm(pos_wrt_base[:2])
                bbox_id2dist[bbox[0]] = dist
            print(f"bbox_id2dist: {bbox_id2dist}")

            info.update({
                'bbox_ignore_ids': [0],
                'bbox_id2dist': bbox_id2dist,
            })
            prompt_rgb, obj_bbox_list = bbox_prompt_img(
                im=rgb.copy(),
                info=info,
                bboxes=bboxes,
                prompt_args=self.prompt_args,
            )
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_img_{info["save_key"]}.png'))

            encoded_image = U.encode_image(prompt_rgb)
            history_msgs = None
            if (history is not None) and (len(history)>0):
                history_msgs = self.create_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                )

            n_retries = 3
            for _ in range(n_retries):
                response = self.vlm_runner(
                    encoded_image=encoded_image,
                    history_msgs=history_msgs,
                    make_prompt_func=make_prompt_func,
                    make_prompt_func_kwargs={
                        'query': query,
                        'info': info,
                    }
                )
                U.save_image(depth.astype(np.uint8), os.path.join(self.vis_dir, f'depth_{info["save_key"]}.png'))
                coord, button_id, return_info = self.get_param_from_response(response, obj_bbox_list=obj_bbox_list, pcd=pcd, mask_image=np.asarray(mask_image))

                capture_history = {
                    'image': prompt_rgb,
                    'query': query,
                    'model_response': button_id,
                    'full_response': response,
                    'button_id': button_id,
                    'model_analysis': '',
                }

                return_info['floor_num'] = info['floor_num']
                self.save_model_output(
                    rgb=prompt_rgb,
                    response=response,
                    subtitles=[f'Task Query: {query}', f'Button ID: {button_id}'],
                    img_file=os.path.join(self.vis_dir, f'output_{info["save_key"]}.png'),
                )
                if coord is not None:
                    break
            if coord is None:
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

            if np.any(np.isnan(pcd[coord[1], coord[0]])):
                # check if anything in the object_mask is not nan
                mask = return_info['object_id_mask']
                pcd_mask = pcd[mask]
                mask = np.all(np.isnan(pcd_mask), axis=1)
                pcd_mask = pcd_mask[~mask]
                if len(pcd_mask) > 0:
                    pos = np.mean(pcd_mask, axis=0)
                    pcd[coord[1], coord[0]] = pos

            # check if pos is not NAN
            if np.any(np.isnan(pcd[coord[1], coord[0]])):
                # ideally this should not happen at all.
                error = "The selected position is not valid due to depth sensor error. Please try again after moving the base."
                return self.on_failure(
                    reason_for_failure=error,
                    reset_required=False,
                    capture_history=capture_history,
                    return_info=return_info,
                )

            pos = pcd[coord[1], coord[0]]
            if (adjust_y) and (bld == 'ahg'):
                pos[1] = pos[1] - 0.045
            if (adjust_y) and (bld == 'mbb'):
                pos[1] = pos[1] - 0.02

            normal = None
            if normals is not None:
                normal = normals[coord[1], coord[0]]

        if self.oracle_position:
            capture_history = {}
            clicked_points = U.get_user_input(rgb)
            assert len(clicked_points) == 1
            print(f'clicked_points: {clicked_points}')
            pos = pcd[clicked_points[0][1], clicked_points[0][0]] # pos of pad in the base_footprint frame
            U.clear_input_buffer()
            floor_num = input("enter the target floor number:")
            return_info = {'floor_num': floor_num}
            normal = None
            if normals is not None:
                normal = normals[clicked_points[0][1], clicked_points[0][0]]

        orig_pos = copy.deepcopy(pos)
        opp_arm = 'right' if arm == 'left' else 'left'
        right_pad_wrt_base = T.pose2mat(self.tf_base.get_transform(f'/gripper_{arm}_{opp_arm}_inner_finger_pad'))
        right_arm_wrt_base = T.pose2mat(self.tf_base.get_transform(f'/arm_{arm}_tool_link'))
        # calculate the pos of the arm_tool_link in the base_footprint frame
        translation = right_pad_wrt_base[:3, 3] - right_arm_wrt_base[:3, 3] - np.asarray([0.0, 0.03, 0.0]) # this is some small offset observed in the real robot
        pos = pos - translation - np.asarray([0.015, 0.0, 0.01]) # this is offset for avoid collision

        frame = 'base_footprint'
        current_arm_pose = self.left_arm_pose(frame=frame) if arm == 'left' else self.right_arm_pose(frame=frame)

        start_arm_pos, start_arm_ori = current_arm_pose[:3], current_arm_pose[3:7]
        approach_pos, approach_ori = self.get_approach_pose(pos, normal, arm=arm, frame=frame)
        goto_pos_base, goto_ori = self.get_goto_pose(pos, normal, arm=arm, approach_ori=approach_ori, frame=frame)
        transform = None # no transformation from base_footprint is required.

        if self.debug:
            pcd_to_plot = pcd.reshape(-1,3)
            # transform it to the global frame
            if transform is not None:
                # pad it with 1.0 for homogeneous coordinates
                pcd_to_plot = np.concatenate((pcd_to_plot, np.ones((pcd_to_plot.shape[0], 1))), axis=1)
                pcd_to_plot = (transform @ pcd_to_plot.T).T
                pcd_to_plot = pcd_to_plot[:, :3]

            # concatenate the approach pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, approach_pos.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0) # approach pose in red
            # # add the orig_pos
            pcd_to_plot = np.concatenate((pcd_to_plot, orig_pos.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 255.0, 0.0]])), axis=0) # orig pos in green
            # # concatenate the goto pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, pos.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 0.0, 255.0]])), axis=0) # goto pose in blue

            U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)

        duration_scale_factor = 3.0
        goto_args = {
            'env': env,
            'arm': arm,
            'frame': frame,
            'gripper_act': None,
            'adj_gripper': False, # we do not adjust the gripper here, since the pos is in tool_link frame
        }

        success = True
        execute = U.confirm_user(execute, "Do you want to continue? (y/n): ", "Pressing the button.")

        if execute:
            print("Moving to the approach pose")
            start_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            obs, reward, done, info = self.arm_goto_pose(pose=(approach_pos, approach_ori), n_steps=2, duration_scale_factor=duration_scale_factor, **goto_args)
            approach_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            print("Moving to the goto pose")
            obs, reward, done, info = self.arm_goto_pose(pose=(goto_pos_base, approach_ori), n_steps=2, force_z_th=-10.0, duration_scale_factor=2*duration_scale_factor, **goto_args)
            print("Moving back to the approach pose")
            # rospy.sleep(2)
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = np.linalg.norm(approach_joint_angles-cur_joint_angles)*duration_scale_factor
            env.tiago.arms[arm].write(approach_joint_angles, duration_scale, delay_scale_factor=duration_scale_factor)
            print("Moving back to the start pose")
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = np.linalg.norm(start_joint_angles-cur_joint_angles)*duration_scale_factor
            env.tiago.arms[arm].write(start_joint_angles, duration_scale, delay_scale_factor=duration_scale_factor)

        if execute:
            self.open_gripper(env=env, side='right')
        return self.on_success(
            capture_history=capture_history,
            return_info=return_info,
        )

class CallElevatorSkill(ElevatorSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_name = "call_elevator"
        self.skill_descs=f"""
skill_name: {self.skill_name}
arguments: button position depending whether you want to go to a floor above or below.
description: Equips the robot with calling the elevator capability. The robot will push the button selected in the argument to call the elevator in the current floor. The subtask must indicate the current floor number and the destination floor number to go to, example, 'Go to the second floor from first floor.'
""".strip()
        run_dir = kwargs.get('run_dir', None)
        self.vis_dir = os.path.join(run_dir, 'call_elevator')
        os.makedirs(self.vis_dir, exist_ok=True)

    def get_history_dirs(self):
        base_dir = os.path.join(bumble.__path__[0], 'long_term_mem', 'call_elevator')
        history_eval_dirs = [os.path.join(base_dir, 'eval_id001.pkl')]
        return history_eval_dirs

    def get_base_approach_pose_map(self, floor_num, bld):
        pos, ori = None, None
        if bld == 'ahg':
            if floor_num == 2:
                pos = np.asarray([-0.60, -8.06, 0.0])
                ori = np.asarray([0.0, 0.0, 0.10280934380435, 0.9947010801374043])
        elif bld == 'mbb':
            if floor_num == 1:
                pos = np.asarray([0.172, 0.236, 0.0])
                ori = np.asarray([0.0, 0.0, 0.0, 1.0])
            elif floor_num == 2:
                pos = np.asarray([-6.373, 4.212, 0.0])
                ori = np.asarray([0.0, 0.0, 0.7133678835846088, 0.7007897421267066])
            elif floor_num == 3:
                raise NotImplementedError
            else: raise NotImplementedError
        return pos, ori

    def get_base_approach_in_pose_map(self, floor_num, bld):
        pos, ori = None, None
        if bld == 'ahg':
            if floor_num == 2:
                pos = np.asarray([-1.28, -7.38, 0.0])
                ori = np.asarray([0.0, 0.0, -0.9964840539559808, 0.08378263669432913])
        elif bld == 'mbb':
            if floor_num == 1:
                pos = np.asarray([0.172, 0.74, 0.0])
                ori = np.asarray([0.0, 0.0, -1.0, 0.0])
            elif floor_num == 2:
                pos = np.asarray([-7.081, 4.213, 0.0])
                ori = np.asarray([0.0, 0.0, -0.7036433446955365, 0.7105533361160712])
            elif floor_num == 3:
                raise NotImplementedError
            else: raise NotImplementedError
        return pos, ori

    def step(self, env, rgb, depth, pcd, normals, arm, query, execute=True, run_vlm=True, info=None, **kwargs):
        '''
            This is an open-loop skill to push a button
            if execute is False, then it will only return the success flag
        '''
        info = copy.deepcopy(info) if info is not None else {}
        pos = None
        assert 'floor_num' in info.keys()
        floor_num = info['floor_num']
        success, reason_for_failure, capture_history, return_info = \
                self.press_once(
                    env=env,
                    rgb=rgb,
                    depth=depth,
                    pcd=pcd, normals=normals,
                    arm=arm,
                    query=query,
                    execute=execute,
                    run_vlm=run_vlm,
                    info=info,
                    close_gripper=True,
                    make_prompt_func=make_prompt,
                    make_history_prompt=make_history_call_elevator,
                    **kwargs
                )
        if not success:
            return success, reason_for_failure, capture_history, return_info

        execute = U.confirm_user(execute, "Do you want to continue? (y/n): ", "Going inside the elevator.")

        if execute:
            duration_scale_factor = 0.5
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            goal_joint_angles = copy.deepcopy(cur_joint_angles)
            goal_joint_angles[-2] = -0.28
            duration_scale = duration_scale_factor*np.linalg.norm(goal_joint_angles-cur_joint_angles) # move twice as low
            env.tiago.arms[arm].write(goal_joint_angles, duration_scale, delay_scale_factor=duration_scale_factor)

            goal = self.create_move_base_goal(self.get_base_approach_pose_map(floor_num, bld=kwargs['bld']))
            state = self.send_move_base_goal(goal)

            goal = self.create_move_base_goal(self.get_base_approach_in_pose_map(floor_num, bld=kwargs['bld']))
            state = self.send_move_base_goal(goal)

            goal_joint_angles = copy.deepcopy(cur_joint_angles)
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = duration_scale_factor*np.linalg.norm(goal_joint_angles-cur_joint_angles) # move twice as low
            env.tiago.arms[arm].write(goal_joint_angles, duration_scale, delay_scale_factor=duration_scale_factor)

        return success, reason_for_failure, capture_history, return_info

class UseElevatorSkill(ElevatorSkill):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_name = "use_elevator"
        self.skill_descs=f"""
skill_name: {self.skill_name}
arguments: button position of the floor to go to.
description: Equips the robot with using elevator capabilities. The robot will push the button selected in the argument. This skill is used to change the floor of the robot after calling the elevator. The subtask must indicate the desired floor number to go to.
""".strip()
        run_dir = kwargs.get('run_dir', None)
        self.vis_dir = os.path.join(run_dir, 'use_elevator')
        os.makedirs(self.vis_dir, exist_ok=True)

    def get_base_approach_pose_map(self, floor_num, bld):
        pos, ori = self.tf_map.get_transform('/base_footprint')
        trasnform = T.pose2mat((pos, ori))
        pos = np.asarray([0.0, 0.50, 0.0]) # move 50 cm to the right in the base_footprint frame
        pos = trasnform @ np.concatenate((pos, np.asarray([1.0])), axis=0)
        pos = pos[:3]
        return pos, ori

    def get_base_approach_out_pose_map(self, floor_num, bld):
        pos, ori = self.tf_map.get_transform('/base_footprint')
        trasnform = T.pose2mat((pos, ori))
        if bld == 'ahg':
            pos = np.asarray([3.0, 0.50, 0.0]) # move 50 cm right and 1m forward in the base_footprint frame
        elif bld == 'mbb':
            pos = np.asarray([2.0, 0.50, 0.0]) # move 50 cm right and 1m forward in the base_footprint frame
        else:
            raise NotImplementedError
        pos = trasnform @ np.concatenate((pos, np.asarray([1.0])), axis=0)
        pos = pos[:3]
        return pos, ori

    def get_history_dirs(self):
        base_dir = os.path.join(bumble.__path__[0], 'long_term_mem', 'call_elevator')
        history_eval_dirs = [os.path.join(base_dir, 'eval_id001.pkl')]
        return history_eval_dirs

    def step(self, env, rgb, depth, pcd, normals, arm, query, execute=True, run_vlm=True, info=None, **kwargs):
        '''
            This is an open-loop skill to push a button
            if execute is False, then it will only return the success flag
        '''
        info = copy.deepcopy(info) if info is not None else {}
        pos = None
        assert 'floor_num' in info.keys()

        floor_num = info['floor_num']

        if env is not None:
            env.reset(reset_arms=False, reset_pose={'torso': 0.35, 'left': None, 'right': None})

            obs_pp = VU.get_obs(env, self.tf_base)
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

        save_key = info['save_key']
        save_key += '_2'
        info.update({'save_key': save_key})

        success, reason_for_failure, capture_history, return_info = \
                self.press_once(
                    env=env,
                    rgb=rgb,
                    depth=depth,
                    pcd=pcd, normals=normals,
                    arm=arm,
                    query=query,
                    execute=execute,
                    run_vlm=run_vlm,
                    info=info,
                    close_gripper=False,
                    make_prompt_func=make_prompt_floor_ch,
                    make_history_prompt=make_history_use_elevator,
                    adjust_y=True,
                    **kwargs
                )

        rospy.sleep(2)
        if success:
            if run_vlm:
                floor_num, return_info = self.get_param_from_response_floor_num(capture_history['full_response'], return_info)
                capture_history['model_response'] = [capture_history['model_response'], floor_num]
            else:
                floor_num = input("Enter the target floor number: ")
                floor_num = int(floor_num)
            if floor_num == '':
                # ask the user to set it!
                U.clear_input_buffer()
                floor_num = int(input("Please enter the current floor number [1,2,3]: "))
            return_info['floor_num'] = int(floor_num)
            print(colored(f"Floor number changed to {return_info['floor_num']}", 'green'))

        execute = U.confirm_user(execute, "Do you want to continue? (y/n): ", "Moving outsidie the elevator.")
        if execute:
            duration_scale_factor = 1.0
            self.tuck_in_gripper(env, arm=arm)

            goal = self.create_move_base_goal(self.get_base_approach_pose_map(return_info['floor_num'], bld=kwargs['bld']))
            state = self.send_move_base_goal(goal)

            goal = self.create_move_base_goal(self.get_base_approach_out_pose_map(return_info['floor_num'], bld=kwargs['bld']))
            state = self.send_move_base_goal(goal)

            self.tuck_out_gripper(env, arm=arm)

        return success, reason_for_failure, capture_history, return_info

