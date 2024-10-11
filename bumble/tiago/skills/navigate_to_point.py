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
from bumble.tiago.prompters.direction import prompt_move_img, prompt_rotate_img
from bumble.tiago.prompters.object_bbox import bbox_prompt_img

from termcolor import colored

def make_prompt_obj(query, info, llm_baseline_info=None, method="ours"):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    if method == "ours":
        visual_instructions = [
            "the image of the scene marked with object id",
            "image",
            "The object_id is the character marked in circle on the object. First, describe all the objects in the scene. Then",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a scene description, descriptions of objects by their ID",
            "Object ID Descriptions section",
            "First",
        ]
    else:
        raise NotImplementedError
    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the object which the robot must move towards to complete the task. You are provided with {visual_instructions[0]}, and the task of the robot. You can ONLY select the object id present in the {visual_instructions[1]}.

You are a five-time world champion in this game. Output only one object id, do NOT leave it empty. {visual_instructions[2]}, give a short analysis of how you would chose the object given the task. Finally, select the object id that the robot must go towards in order to complete the task, in a valid JSON of this format:
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

class NavigateToPointSkill(SkillBase):
    def __init__(
            self,
            oracle_action=False,
            debug=False,
            run_dir=None,
            prompt_args=None,
            skip_ros=False,
            *args, **kwargs
        ):
        # NOTE: We assume that the head is looking straight ahead
        super().__init__(*args, **kwargs)
        self.oracle_action = oracle_action
        self.debug = debug
        if not skip_ros:
            self.setup_listeners()
        self.vis_dir = os.path.join(run_dir, 'navigate_to_point_on_ground')
        os.makedirs(self.vis_dir, exist_ok=True)
        self.add_obj_id = prompt_args.get('add_obj_id', True)
        arrow_length_per_pixel = prompt_args.get('arrow_length_per_pixel', 0.15)
        radius_per_pixel = prompt_args.get('radius_per_pixel', 0.04)
        plot_direction = prompt_args.get('plot_direction', True)
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
            'add_object_boundary': prompt_args.get('add_object_boundary', False),
            'plot_direction': plot_direction,
        }
        self.skill_name = "navigate_to_point_on_ground"
        self.skill_descs = f"""
skill_name: {self.skill_name}
arguments: object
description: Moves the robot to a point near the selected object. This skill can be used to move to a point in the room to perform a task, example, navigating near the toaster to make a toast.
""".strip()

    def get_param_from_response_obj(self, response, obj_bbox_list, pcd, mask_image):
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

        bbox = bbox_selected[0].bbox
        bbox_env_id = bbox_selected[0].env_id
        poses = self.extract_pcd_from_env_id(pcd, mask_image, bbox_env_id, filter_nan=True)
        mean_pos = np.mean(poses, axis=0)
        # if nan, then we need to find points that are not nan and are near the object.
        return mean_pos, object_id, return_info

    def sample_candidate_pts(self, pcd, rgb, fav_pos):
        pcd_ground = pcd.copy()
        rgb_ground = rgb.copy()
        # U.plotly_draw_3d_pcd(pcd_ground.reshape(-1,3), rgb_ground.reshape(-1, 3))
        mask_ground = pcd[:, :, 2] < 0.05
        mask_interest = np.linalg.norm(pcd[:, :, :2] - fav_pos.reshape(-1,3)[:, :2], axis=-1) < 1.0
        _mask  = np.logical_and(mask_ground, mask_interest)
        pcd_ground = pcd_ground[_mask]
        rgb_ground = rgb_ground[_mask]
        pcd_ground, rgb_ground = U.remove_nan_pcd(pcd_ground, rgb_ground)

        _mask = np.zeros_like((pcd[:, :, 2]))
        min_z_non_ground = 0.10; max_z_non_ground = 0.20
        pt_th = 1000
        while np.sum(_mask) < pt_th: # atleast 100 points should be present
            max_z_non_ground += 0.10
            max_z_mask = np.logical_and((pcd[:, :, 2] < max_z_non_ground), (pcd[:, :, 2] > min_z_non_ground))
            mask_interest = np.linalg.norm(pcd[:, :, :2] - fav_pos.reshape(-1,3)[:, :2], axis=-1) < 1.0
            _mask = np.logical_and(mask_interest, max_z_mask)
            if max_z_non_ground > 3.0:
                print("Could not find enough points near the objects that are non-ground.")
                break

        pcd_non_ground = pcd.copy()[_mask]
        rgb_non_ground = rgb.copy()[_mask]

        # keep only contours in _mask
        _mask = U.create_border_mask(_mask)
        pcd_non_ground = pcd.copy()[_mask]
        rgb_non_ground = rgb.copy()[_mask]
        # refiltering the points to remove the points that are outliers created due the border mask
        mask = pcd_non_ground[:, 2] > min_z_non_ground
        pcd_non_ground = pcd_non_ground[mask]
        rgb_non_ground = rgb_non_ground[mask]
        pcd_non_ground, rgb_non_ground = U.remove_nan_pcd(pcd_non_ground, rgb_non_ground)

        print("before the distance calculation")
        print(pcd_ground.shape, pcd_non_ground.shape)
        if pcd_ground.shape[0] == 0:
            print("No ground points found near the object.")
            return None
        bs1 = 32768
        dist = np.zeros((pcd_ground.shape[0], pcd_non_ground.shape[0]), dtype=np.float32)
        for i in range(0, pcd_ground.shape[0], bs1):
            dist[i:i+bs1] = np.linalg.norm(np.expand_dims(pcd_ground[i:i+bs1,:2], axis=1) - np.expand_dims(pcd_non_ground[:,:2], axis=0), axis=-1)
        print("after the distance calculation")
        keep_dist = 0.40
        mask = np.all(dist > keep_dist, axis=1)
        if np.sum(mask) == 0:
            print("No points found that are atleast 40 cm away from the non-ground points. Max distance found: ", np.max(dist))
            for min_th in np.arange(keep_dist, 0.25, -0.01):
                mask = np.all(dist > min_th, axis=1)
                if np.sum(mask) > 0:
                    print("Minimum distance found: ", min_th)
                    break
            if np.sum(mask) == 0:
                print("No points found that are atleast 25 cm away from the non-ground points.")
                print("Minimum distance found: ", np.min(dist))
                return None

        pcd_ground = pcd_ground[mask]
        rgb_ground = rgb_ground[mask]
        print("Ground points after distance thresholding: ", pcd_ground.shape)
        # valid ground points are the points that have x value less than fav_pos x value
        mask = pcd_ground[:, 0] < fav_pos[0]
        pcd_ground = pcd_ground[mask]
        rgb_ground = rgb_ground[mask]
        print("Ground points in front of object: ", pcd_ground.shape)
        if pcd_ground.shape[0] == 0:
            print("No points found in front of the object.")
            return None

        # select the point that is at the minimum distance from the object
        dist = np.linalg.norm(pcd_ground[:,:2] - fav_pos[:2], axis=-1)
        min_dist = np.min(dist, axis=-1)
        i = np.argmin(dist)
        pt = pcd_ground[i]
        min_dist_mask = np.logical_and(dist < min_dist + 0.03, dist > min_dist - 0.03)
        pcd_ground = pcd_ground[min_dist_mask]
        rgb_ground = rgb_ground[min_dist_mask]

        selected_pts = pcd_ground

        selected_pts = [np.mean(selected_pts, axis=0)]
        return selected_pts

    def step(self, env, rgb, depth, pcd, normals, query, execute=True, run_vlm=True, info=None, bboxes=None, mask_image=None, **kwargs):
        '''
            action: Position, Quaternion (xyzw) of the goal
        '''
        print("MoveToSkill: Move to the initial position")
        # get the current position of the robot_base w.r.t. odom
        if execute:
            assert env is not None, "Environment is required to execute the skill"
        cur_pos, cur_ori = None, None
        if env is not None:
            self.send_head_command(head_positions=[0.0, -0.6])
            obs_pp = VU.get_obs(env, self.tf_base)
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
            cur_pos = env.tiago.base.odom_listener.get_most_recent_msg()['pose'][:3]
            cur_ori = env.tiago.base.odom_listener.get_most_recent_msg()['pose'][3:]
        text_direction = ''
        if self.oracle_action:
            # ask the user for direction
            direction = input("Enter the direction (F, B, L, R)")
            if direction == 'F':
                text_direction = 'forward'
            elif direction == 'B':
                text_direction = 'backward'
            elif direction == 'L':
                text_direction = 'left'
            elif direction == 'R':
                text_direction = 'right'
            else:
                print("Invalid direction")
                return False
        if run_vlm:
            info_cp = copy.deepcopy(info)
            prompt_rgb = None
            prompt_rgb = rgb.copy()
            img_size = min(rgb.shape[0], rgb.shape[1])

            gsam_query = ['all objects']
            if (bboxes is None) or (mask_image is None):
                bboxes, mask_image = self.get_object_bboxes(rgb, query=gsam_query)
            if len(bboxes) == 0:
                # this should not happen
                import ipdb; ipdb.set_trace()
                error = "No objects found in the scene."
                return self.on_failure(
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
            # save the overlay image for debugging
            U.save_image(overlay_image, os.path.join(self.vis_dir, f'overlay_image_{info["save_key"]}.png'))
            bbox_id2dist = {}
            for bbox in bboxes:
                center = (bbox[1] + bbox[3]) // 2, (bbox[2] + bbox[4]) // 2
                pos_wrt_base = pcd[center[1], center[0]]
                dist = np.linalg.norm(pos_wrt_base[:2])
                bbox_id2dist[bbox[0]] = dist

            info_cp.update({
                'bbox_ignore_ids': [0],
                'bbox_id2dist': bbox_id2dist,
            })
            prompt_args = copy.deepcopy(self.prompt_args)
            prompt_args.update({
                'radius': int(img_size * 0.03),
                'fontsize': int(img_size * 30 * 0.03),
                'start_point': (rgb.shape[1]//2, rgb.shape[0] - 20),
            })

            prompt_rgb, obj_bbox_list = bbox_prompt_img(
                im=rgb.copy(),
                info=info_cp,
                bboxes=bboxes,
                prompt_args=prompt_args,
            )
            info_cp['obj_bbox_list'] = obj_bbox_list
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_img_{info["save_key"]}.png'))

            encoded_image = U.encode_image(prompt_rgb)
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=None,
                make_prompt_func=make_prompt_obj,
                make_prompt_func_kwargs={
                    'query': query,
                    'info': info,
                }
            )
            fav_pos, object_id, return_info = self.get_param_from_response_obj(response, obj_bbox_list=obj_bbox_list, pcd=pcd, mask_image=np.asarray(mask_image))
            print(f"Object ID: {object_id}")
            print(f"Object Position: {fav_pos}")
            if fav_pos is None:
                return self.on_failure(
                    reason_for_failure="Failed to detect any such object in the scene.",
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )

            selected_pts = self.sample_candidate_pts(pcd, rgb, fav_pos)
            if selected_pts is None:
                return self.on_failure(
                    reason_for_failure="No points found near the object on the ground.",
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )
            assert len(selected_pts) == 1, "Only one point should be selected. Removed the extra points."
            start_pt = selected_pts[0]
            target_pt = fav_pos
            target_quat = VU.look_at_rotate_z(start_pt[:2], target_pt[:2])
            target_pos = start_pt
        else:
            prompt_rgb = rgb.copy()
            response = ''
            return_info = {
                'response': response,
                'model_out': text_direction,
                'error_list': [],
            }

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
            subtitles=[f'Task Query: {query}', f'Direction: {text_direction}'],
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


        transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
        # create the matrix for the target position using target_pos and target_quat
        target_pose_base = T.pose2mat((target_pos, target_quat))
        target_pose_map = (transform @ target_pose_base)
        goal_pos_map = target_pose_map[:3, 3]
        goal_ori_map = T.mat2quat(target_pose_map[:3, :3])
        print(f"Goal pos in map: {goal_pos_map}")
        print(f"Goal ori in map: {goal_ori_map}")

        if self.debug:
            pcd_wrt_map = np.concatenate((pcd.reshape(-1, 3), np.ones((pcd.reshape(-1,3).shape[0], 1))), axis=1)
            transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
            pcd_wrt_map = (transform @ pcd_wrt_map.T).T
            pcd_wrt_map = pcd_wrt_map[:, :3]
            pcd_to_plot = pcd_wrt_map.reshape(-1,3)
            rgb_to_plot = rgb.reshape(-1,3)

            pcd_to_plot = np.concatenate((pcd_to_plot, goal_pos_map.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0) # goal pos in red

            U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)

        is_success = False
        error = None
        execute = U.confirm_user(execute, "Do you want to continue? (y/n): ", f"Using the navigate_to_point_on_ground skill for {text_direction} direction")
        if execute:
            goal = self.create_move_base_goal((goal_pos_map, goal_ori_map))
            state = self.send_move_base_goal(goal)
            if state == GoalStatus.SUCCEEDED:
                is_success = True
            else:
                error = movebase_code2error(state)
                is_success = False

        if env is not None:
            self.send_head_command(head_positions=self.default_head_joint_position)

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
