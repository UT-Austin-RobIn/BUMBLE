#!/usr/bin/env python3
import os
import sys
import copy
import time
import numpy as np
from math import pi
from scipy.spatial.transform import Rotation as R

import rospy
import moveit_commander
import moveit_msgs.msg
from control_msgs.msg import JointTrajectoryControllerState
import geometry_msgs.msg
from std_msgs.msg import String

from bumble.tiago.skills.base import SkillBase
import bumble.utils.utils as U
import bumble.utils.transform_utils as T # transform_utils
from bumble.tiago.prompters.object_bbox import bbox_prompt_img

def make_prompt(query, info, llm_baseline_info=None, method="ours"):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    if method == "ours":
        visual_instructions = [
            "the image of the scene marked with object id",
            "image",
            "The object_id is the character marked in circle next to the object. ",
            "describe all the objects in the scene. Then, ",
            " id"
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a description of the scene, a description of the objects by their id",
            "object descriptions",
            "",
            "",
            " id",
        ]
    elif method == "ours_no_markers":
        visual_instructions = [
            "the image of the scene",
            "image",
            "",
            "describe all the objects in the scene. Then, ",
            "",
        ]
    else:
        raise NotImplementedError
    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the object{visual_instructions[4]} that the robot must pick up to complete the task. You are provided with {visual_instructions[0]}, and the task of the robot. You can ONLY select the object{visual_instructions[4]} present in the {visual_instructions[1]}. The object{visual_instructions[4]}s are NOT serially ordered.

You are a five-time world champion in this game. Output only one object{visual_instructions[4]}, do NOT leave it empty. {visual_instructions[2]}First, {visual_instructions[3]}give a short analysis of how you would chose the object. Then, select object that must be picked up to complete the task. Finally, provide the object{visual_instructions[4]} that must be picked up in a valid JSON of this format:
{{"object_id": ""}}
"""

    if llm_baseline_info:
        instructions += f"""\n
SCENE DESCRIPTION:
{llm_baseline_info['im_scene_desc']}
OBJECT ID DESCRIPTIONS:
{llm_baseline_info['obj_descs']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {query}"""
    task_prompt += f"""
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt


class PickupSkill(SkillBase):
    def __init__(
            self,
            oracle_position: bool = False,
            use_vlm: bool = True,
            adjust_gripper: bool = True,
            debug: bool = False,
            run_dir: str = None,
            prompt_args: dict = None,
            z_offset: float = 0.02,
            skip_ros: bool = False,
            *args, **kwargs,
        ):
        super().__init__(*args, **kwargs)
        # it will prompt the user to click on the object
        self.oracle_position = oracle_position
        # it will use the VLM to predict the object id
        self.use_vlm = use_vlm
        self.skip_ros = skip_ros
        if not self.skip_ros:
            self.setup_listeners()
        self.z_offset = z_offset
        self.adjust_gripper = adjust_gripper
        self.adjust_gripper_length = 0.21 # 85
        self.approach_vec_base = np.asarray([0.0, 0.0, 0.05])
        self.debug = debug
        self.vis_dir = os.path.join(run_dir, 'pickup')
        os.makedirs(self.vis_dir, exist_ok=True)
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
        }

        self.skill_name = "pick_up_object"
        self.skill_descs=f"""
skill_name: {self.skill_name}
arguments: object_of_interest
description: pick_up_object skill moves its arms to pick up the object specified in the argument object_of_interest. The pick_up_object skill can only pick up objects within the reach of its arms and does not control the robot base. The robot cannot pick up heavy objects like chairs, tables, etc.
""".strip()

    def get_param_from_response(self, response, obj_bbox_list, pcd, mask_image, execute):
        '''
            skill_specific function to get the param from the vlm response
        '''
        return_info = {}
        return_info['response'] = response
        return_info['error_list'] = []
        object_id = ''
        try:
            object_id = U.extract_json(response, 'object_id')
            print(f"Object ID: {object_id}")
        except Exception as e:
            print(str(e))
            object_id = ''
        return_info['object_id'] = object_id
        bbox_selected = [bbox for bbox in obj_bbox_list if bbox.obj_id.lower() == object_id.lower()]
        if len(bbox_selected) == 0:
            error = f"Object id {object_id} not found in the scene."
            return_info['error_list'].append(error)
            return None, object_id, return_info
        if not execute:
            return None, object_id, return_info

        bbox = bbox_selected[0].bbox
        bbox_env_id = bbox_selected[0].env_id
        mask = mask_image == bbox_env_id
        poses = pcd[mask].reshape(-1, 3)
        # remove the points that have nan values
        mask = np.all(np.isnan(poses), axis=1)
        poses = poses[~mask]
        if len(poses) == 0:
            error = f"Couldn't grasp the object."
            return_info['error_list'].append(error)
            return None, object_id, return_info
        # highest point (z) is the object
        highest_z_val = np.max(poses[:, 2])
        # x value should be the minimum x value of the object
        x_val = np.min(poses[:, 0])
        # y value should be the average of min and max y value of the object
        # take ones those y values whose z value is > min_z_val by 1cm
        valid_y = poses[poses[:, 2] > np.min(poses[:, 2]) + 0.01]
        if len(valid_y) == 0:
            valid_y = poses
        y_val = np.mean([np.min(valid_y[:, 1]), np.max(valid_y[:, 1])])
        mean_pos = np.asarray([x_val, y_val, highest_z_val])

        return_info['bbox'] = bbox
        coord = (int((bbox[1]+bbox[3])/2.0), int((bbox[2]+bbox[4])/2.0))
        return_info['coord'] = coord
        return mean_pos, object_id, return_info

    def get_approach_pose(self, pos, normal, frame='odom', current_arm_pose=None):
        '''
            pos, normal: np.ndarray are w.r.t. the base_footprint
        '''
        assert frame in ['base_footprint']
        approach_pos = pos + self.approach_vec_base
        approach_ori = current_arm_pose[3:7]
        approach_ori = np.asarray([0.0, -0.717, 0.0, -0.717])
        # convert it to a valid quaternion
        approach_ori /= np.linalg.norm(approach_ori)
        return approach_pos, approach_ori

    def get_goto_pose(self, pos, normal, frame, info):
        assert frame in ['odom', 'base_footprint']
        approach_pos = info['approach_pos']
        approach_ori = info['approach_ori']

        goto_pos = pos - np.asarray([0.0, 0.0, self.z_offset]) # go down by 2cm
        goto_ori = approach_ori.copy()
        return goto_pos, goto_ori

    def on_failure(self, env, arm, *args, **kwargs):
        '''
            This function is called when the skill fails
        '''
        if env:
            self.open_gripper(env, arm)
        return super().on_failure(*args, **kwargs)

    def step(self, env, rgb, depth, pcd, normals, arm, query, execute=True, run_vlm=True, info=None, bboxes=None, mask_image=None, **kwargs):
        '''
            This is an open-loop skill to push a button
            if execute is False, then it will only return the success flag
        '''
        info = copy.deepcopy(info) if info is not None else {}
        pos = None
        clicked_points = []
        pos, normal = None, None
        if run_vlm:
            assert self.use_vlm
            # gsam_query = ['food items', 'snacks', 'drinks']
            gsam_query = ['all objects']
            if (bboxes is None) or (mask_image is None):
                bboxes, mask_image = self.get_object_bboxes(rgb, query=gsam_query)
            if len(bboxes) == 0:
                # this should not happen
                import ipdb; ipdb.set_trace()
                error = "No objects found in the scene."
                self.on_failure(
                    env=env,
                    arm=arm,
                    reason_for_failure=error,
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )
            # used mainly for debugging
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
                if pcd is not None:
                    center = (bbox[1] + bbox[3]) // 2, (bbox[2] + bbox[4]) // 2
                    pos_wrt_base = pcd[center[1], center[0]]
                    dist = np.linalg.norm(pos_wrt_base[:2])
                else:
                    bbox_id2dist[bbox[0]] = 0.0
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
            if self.method == "ours_no_markers":
                prompt_rgb = rgb.copy()
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_img_{info["save_key"]}.png'))

            encoded_image = U.encode_image(prompt_rgb)
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=None,
                make_prompt_func=make_prompt,
                make_prompt_func_kwargs={
                    'query': query,
                    'info': info,
                }
            )
            if depth is not None:
                U.save_image(depth.astype(np.uint8), os.path.join(self.vis_dir, f'depth_{info["save_key"]}.png'))
            pos, object_id, return_info = self.get_param_from_response(response, obj_bbox_list=obj_bbox_list, pcd=pcd, mask_image=np.asarray(mask_image), execute=execute)
        else:
            prompt_rgb = rgb.copy()
            response = ''
            object_id = 0
            return_info = {
                'response': response,
                'model_out': object_id,
                'error_list': [],
            }


        capture_history = {
            'image': prompt_rgb,
            'query': query,
            'model_response': object_id,
            'full_response': response,
            'object_id': object_id,
            'model_analysis': '', # this will be added by an external evaluator
        }
        self.save_model_output(
            rgb=prompt_rgb,
            response=response,
            subtitles=[f'Task Query: {query}', f'Object ID: {object_id}'],
            img_file=os.path.join(self.vis_dir, f'output_{info["save_key"]}.png'),
        )

        error = None
        if len(return_info['error_list']) > 0:
            error = "Following errors have been produced: "
            for e in return_info['error_list']:
                error += f"{e}, "
            error = error[:-2]
            return self.on_failure(
                env=env,
                arm=arm,
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        if self.oracle_position:
            clicked_points = U.get_user_input(rgb)
            assert len(clicked_points) == 1
            print(f'clicked_points: {clicked_points}')
            pos = pcd[clicked_points[0][1], clicked_points[0][0]]
            normal = normals[clicked_points[0][1], clicked_points[0][0]]

        # ========================== Transform the pose specified for gripper_{arm}_{arm}_inner_finger_pad to a position for {arm}_tool_link ==========================
        frame = 'base_footprint'
        if not self.skip_ros:
            arm_pad_wrt_base = T.pose2mat(self.tf_base.get_transform(f'/gripper_{arm}_{arm}_inner_finger_pad'))
            arm_wrt_base = T.pose2mat(self.tf_base.get_transform(f'/arm_{arm}_tool_link'))
            translation = arm_pad_wrt_base[:3, 3] - arm_wrt_base[:3, 3] - np.asarray([0.0, 0.03, 0.0]) #- np.asarray([0.0, 0.01, 0.0]) # any offset
            orig_pos = copy.deepcopy(pos)
            pos = pos - translation
            pos = pos - np.asarray([0.02, 0.0, 0.0]) # maintain an offset of 1cm from the object

            # current_arm_pose in base_footprint
            current_arm_pose = self.left_arm_pose(frame=frame) if arm == 'left' else self.right_arm_pose(frame=frame)

            start_arm_pos, start_arm_ori = current_arm_pose[:3], current_arm_pose[3:7]
            approach_pos, approach_ori = self.get_approach_pose(pos, normal, frame=frame, current_arm_pose=current_arm_pose)

            goto_pos_base, goto_ori_base = self.get_goto_pose(pos, normal, frame=frame, info={'approach_pos': approach_pos, 'approach_ori': approach_ori})
            transform = None # no transformation from base_footprint is required.

        if self.debug:
            pcd_to_plot = pcd.reshape(-1,3)
            # transform it to the global frame
            if transform is not None:
                # pad it with 1.0 for homogeneous coordinates
                pcd_to_plot = np.concatenate((pcd_to_plot, np.ones((pcd_to_plot.shape[0], 1))), axis=1)
                pcd_to_plot = (transform @ pcd_to_plot.T).T
                pcd_to_plot = pcd_to_plot[:, :3]
            rgb_to_plot = rgb.reshape(-1,3)

            # concatenate the approach pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, approach_pos.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0) # approach pose in red

            # concatenate the goto pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, goto_pos_base.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 0.0, 255.0]])), axis=0) # goto pose in blue

            # concatenate the current arm pose to the pcd
            pcd_to_plot = np.concatenate((pcd_to_plot, current_arm_pose[:3].reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[128.0, 128.0, 128.0]])), axis=0) # current arm pose in green

            pcd_to_plot = np.concatenate((pcd_to_plot, orig_pos.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot, np.asarray([[0.0, 255.0, 0.0]])), axis=0) # object pose in green

            U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)

        success = True
        error = None
        goto_args = {
            'env': env,
            'arm': arm,
            'frame': frame,
            'gripper_act': None,
            'adj_gripper': False,
        }

        execute = U.confirm_user(execute, "Do you want to continue? (y/n): ", "Picking up the object.")
        if not execute:
            return self.on_failure(
                env=env,
                arm=arm,
                reason_for_failure="Failed: IK solver failed to find the solution for the goto pose.",
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )
        if execute:
            # ========================== Grasping the object ==========================
            duration_scale_factor = 3.0
            start_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            print("Moving to the approach pose")
            obs, reward, done, info = self.arm_goto_pose(pose=(approach_pos, approach_ori), n_steps=2, duration_scale_factor=duration_scale_factor, **goto_args)

            approach_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            print("Moving to the goto pose")
            obs, reward, done, info = self.arm_goto_pose(pose=(goto_pos_base, goto_ori_base), n_steps=2, duration_scale_factor=2*duration_scale_factor, **goto_args) # move twice as slow
            if info[f'arm_{arm}']['joint_goal'] is None:
                # joint_goal is None, this means that the IK solver failed to find the solution
                error = "IK solver failed to find the solution for the goto pose."
                success = False

            print("Closing the gripper")
            self.close_gripper(env, arm)

            # ========================== Return to the start pose ==========================
            print("Moving back to the approach pose")
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = 2*duration_scale_factor*np.linalg.norm(approach_joint_angles-cur_joint_angles) # move twice as low
            env.tiago.arms[arm].write(approach_joint_angles, duration_scale, delay_scale_factor=duration_scale_factor)
            print("Moving back to the start pose")
            cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
            duration_scale = duration_scale_factor * np.linalg.norm(start_joint_angles-cur_joint_angles)
            env.tiago.arms[arm].write(start_joint_angles, duration_scale, delay_scale_factor=duration_scale_factor)
            # NOTE: the error in skill execution failing is NOT being captured here. Skill is open-loop!

        if not success:
            return self.on_failure(
                env=env,
                arm=arm,
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        return self.on_success(
            capture_history=capture_history,
            return_info=return_info,
        )
