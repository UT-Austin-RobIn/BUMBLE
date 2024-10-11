import os
import sys
import copy
import pickle
import numpy as np

import rospy
import actionlib
# from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

import bumble
from bumble.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from bumble.tiago.skills.base import SkillBase, movebase_code2error
import bumble.utils.utils as U
import bumble.utils.vision_utils as VU # vision_utils
import bumble.utils.transform_utils as T # transform_utils
import bumble.tiago.RESET_POSES as RP
from bumble.tiago.prompters.direction import prompt_move_img, prompt_rotate_img
from bumble.tiago.prompters.object_bbox import bbox_prompt_img
from bumble.tiago.ros_restrict import change_map

from termcolor import colored
from matplotlib import pyplot as plt

def make_prompt_object(query, info, llm_baseline_info=None, method="ours"):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    if method == "ours":
        visual_instructions = [
            "the image of the scene marked with object id",
            "image",
            " The object_id is the character marked in circle next to the object.",
            "describe all the objects in the scene. Then, ",
            " id",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a description of the scene and the objects on the scene by their id",
            "object descriptions section",
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
            ""
        ]
    else:
        raise NotImplementedError
    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the object robot must push to complete the task. You are provided with {visual_instructions[0]}, and the task of the robot. You can ONLY select an object{visual_instructions[4]} present in the {visual_instructions[1]}. All the objects are blocking the path of the robot. Avoid pushing objects that are delicate or can cause accidents later, example, stop sign. The robot can push objects that are not directly in the path but can be pushed to clear the path.

You are a five-time world champion in this game. Output only one object{visual_instructions[4]}, do NOT leave it empty.{visual_instructions[2]}
First, summarize all the errors made in previous predictions if provided. Then, {visual_instructions[3]}describe which objects should NOT be pushed and why. Then, select one object that can be pushed to complete the task among the options. Finally, provide the object{visual_instructions[4]} that must be pushed in a valid JSON of this format:
{{"object_id": ""}}
""".strip()

    if llm_baseline_info:
        instructions += f"""\n
SCENE DESCRIPTION:
{llm_baseline_info['im_scene_desc']}
OBJECT ID DESCRIPTIONS:
{llm_baseline_info['obj_descs']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {query}"""
    task_prompt += f"""\n
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

def make_prompt_dir(query, info, llm_baseline_info=None, method="ours"):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    if method == "ours":
        visual_instructions = [
            "an image of the scene marked with the object id along with the three directions that the robot can push the object",
            " marked by 'F', 'L', and 'R' in the image provided",
            "The forward direction is moving in the direction of the image (towards the top of the image), the left direction is moving to the left of the image, and right is moving to the right of the image.",
            "Then, describe the scene in image. ",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a description of the scene with the object id",
            "",
            "The forward direction is moving toward the objects on the scene, the left direction is moving to the left of the scene, and right is moving to the right of the scene.",
            "First, ",
        ]
    elif method == "ours_no_markers":
        visual_instructions = [
            "an image of the scene",
            "",
            "The forward direction is moving in the direction of the image (towards the top of the image), the left direction is moving to the left of the image, and right is moving to the right of the image.",
            "First, describe the scene in image. ",
        ]
    else:
        raise NotImplementedError
    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the direction in which the robot must push the object complete the task. You are provided with a description of the task, and {visual_instructions[0]}. The robot can push in ONLY ONE of the three directions: forward, left, or right{visual_instructions[1]}. {visual_instructions[2]}

You are a five-time world champion in this game. Output only one of the directions: forward, left, or right. Do NOT leave it empty. First, summarize all the errors made in the prediction history if provided. If the previous errors are not provided, specify that no prediction failures are provided. {visual_instructions[3]}Summarize what is present in all the directions of the selected object. Then, describe the task and provide a short analysis of how you would chose the direction to push the object marked. Check if all the conditions for pushing are satisfied. Then, select the direction that can best help complete the task. Finally, provide the direction in a valid JSON of this format:
{{"direction_to_push": ""}}

GUIDELINES:
    - If you push the object to the right, the right side of the object should be empty for the robot to push.
    - If you push the object to the left, the left side of the object should be empty for the robot to push.
    - If you push the object forward, the path in front of the object should be empty for the robot to push. Check for any walls or tables in front of the object. If the task is to clear pathway for the robot, it can push the object in the front and move sideways after pushing if both sides are blocked.
""".strip()

    if llm_baseline_info:
        instructions += f"""\n
SCENE DESCRIPTION:
{llm_baseline_info['im_scene_desc']}
OBJECT ID DESCRIPTIONS:
{llm_baseline_info['obj_descs']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {query}"""
    task_prompt += f"""\n
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

def make_history_prompt(history):
    instructions = f"""
Below is some of the prediction history that can help you understand the mistakes and successful predictions made in the past. Pay close attention to the mistakes made in the past and try to avoid them in the current prediction. Summarize each of prediction failures in the past. Based on the history, improve your prediction. Note that the object ids are marked differently in each image.
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
- Answer: {{'object_id': {msg['model_response'][0]}}}
""".strip()
        history_desc.append(example_desc)
        history_model_analysis.append(msg['model_analysis'])
    return instructions, history_desc, history_model_analysis

def make_history_prompt_dir(history):
    instructions = f"""
Below is some of the prediction history that can help you understand the mistakes and successful predictions made in the past. Pay close attention to the mistakes made in the past and try to avoid them in the current prediction. Summarize each of prediction failures in the past. Based on the history, improve your prediction. Note that the object ids are marked differently in each image.
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
- Answer: {{'direction_to_push': {msg['model_response'][1]}}}
""".strip()
        history_desc.append(example_desc)
        history_model_analysis.append(msg['model_analysis'])
    return instructions, history_desc, history_model_analysis

class PushObsGrSkill(SkillBase):
    # pushes obstacle on the ground
    def __init__(
            self,
            oracle_action=False,
            debug=False,
            run_dir=None,
            prompt_args=None,
            skip_ros=False,
            add_histories=False,
            *args, **kwargs
        ):
        # NOTE: We assume that the head is looking straight ahead
        super().__init__(*args, **kwargs)
        move_dist = 0.5
        self.move_left_length = move_dist
        self.move_forward_length = move_dist
        self.move_right_length = move_dist
        self.add_histories = add_histories

        self.oracle_action = oracle_action
        self.debug = debug
        if not skip_ros:
            self.setup_listeners()
        self.vis_dir = os.path.join(run_dir, 'push_object_on_ground')
        os.makedirs(self.vis_dir, exist_ok=True)

        arrow_length_per_pixel = prompt_args.get('arrow_length_per_pixel', 0.15)
        radius_per_pixel = prompt_args.get('radius_per_pixel', 0.06)
        self.add_obj_id = prompt_args.get('add_obj_id', True)
        if self.method != "ours":
            plot_direction = False
        else:
            plot_direction = prompt_args.pop('plot_direction', True)
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
            'add_object_boundary': prompt_args.pop('add_object_boundary', False),
            'plot_direction': plot_direction,
            'skill_type': 'move', # this can be considered as move skill
            'move_dist': move_dist,
        }
        self.skill_name = "push_object_on_ground"
        self.skill_descs = f"""
skill_name: {self.skill_name}
arguments: object, direction
description: Pushes an object on the ground one of the directions. The robot can push objects that are further away from the robot within two-three meters. The skill decides which object, direction to push, example use case, pushing obstacle to clear the pathway for the robot to go forward, pushing objects for rearrangement, etc.
""".strip()
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
                _history_list = [h for h in _history_list if not h['is_success']]
                _history_list = _history_list[:samples_per_hist]
                history_list.extend(_history_list)
            self.history_list = history_list
            print(f"Loaded {len(history_list)} failed samples.")

    def get_history_dirs(self):
        base_dir = os.path.join(bumble.__path__[0], 'long_term_mem', 'push_obs_on_ground')
        history_eval_dirs = [os.path.join(base_dir, 'eval_id001.pkl'), os.path.join(base_dir, 'eval_id004.pkl')]
        return history_eval_dirs

    def push_pull_arm(self, env, arm, goto_pos=None):
        # move the right arm to push the object
        # this function set the right arm pose to push the object with 20cm, 20cm, 20cm, 10cm
        cur_arm_base = self.right_arm_pose('base_footprint') if arm == 'right' else self.left_arm_pose('base_footprint')
        start_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
        tar_joint_angles = [1.50, 0.189, 1.010, 0.206, 0.972, -0.064, 1.60]
        duration_scale = np.linalg.norm(tar_joint_angles-start_joint_angles) # move twice as low
        env.tiago.arms[arm].write(tar_joint_angles, duration_scale, delay_scale_factor=1.0)
        if goto_pos is not None: # extend the arm and goto the desired position
            # goto_pos must be in odom
            success = self.goto_odom_pos(env, goto_pos)
        env.tiago.arms[arm].write(start_joint_angles, duration_scale, delay_scale_factor=1.0)
        return True

    def get_bbox_from_response(self, response, obj_bbox_list, info):
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
        return_info['model_out'] = object_id
        return_info['object_id'] = object_id
        bbox_selected = [bbox for bbox in obj_bbox_list if bbox.obj_id.lower() == object_id.lower()]
        if len(bbox_selected) == 0:
            error = f"Object id {object_id} not found in the scene."
            return_info['error_list'].append(error)
            return None, object_id, return_info
        return bbox_selected, object_id, return_info

    def get_dir_from_response(self, response, info):
        return_info = {}
        return_info['response'] = response
        direction = ''
        error_list = []
        try:
            direction = U.extract_json(response, 'direction_to_push')
            print(f"Direction: {direction}")
            if direction.lower() in ['r', 'l', 'f']:
                if direction.lower() == 'r':
                    direction = 'right'
                elif direction.lower() == 'l':
                    direction = 'left'
                elif direction.lower() == 'f':
                    direction = 'forward'
            if direction.lower() not in ['forward', 'left', 'right']:
                error = 'Invalid direction. Please provide one of the following directions: forward, left, right.'
                error_list.append(error)
                direction = ''
        except Exception as e:
            print(f"Error: {e}")
            error = 'Invalid response format. Please provide the direction in a valid JSON format.'
            error_list.append(error)
            direction = None
        return_info['model_out'] = direction
        return_info['text_direction'] = direction
        return_info['error_list'] = error_list
        return direction, return_info

    def move_behind(self, env):
        env.reset(reset_arms=True, reset_pose=RP.HOME_L_HOME_R_H)
        base_goal_pose = T.pose2mat((np.asarray([-1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0, 1.0])))
        transform = T.pose2mat(self.tf_map.get_transform('/base_footprint'))
        goal_pose_map = transform @ base_goal_pose
        goal_pose_map = T.mat2pose(goal_pose_map)
        goal = self.create_move_base_goal(goal_pose_map)
        state = self.send_move_base_goal(goal)
        env.reset(reset_arms=False, reset_pose={'torso': 0.35, 'left': None, 'right': None})
        return True

    def step(self, env, rgb, depth, pcd, normals, query, execute=True, run_vlm=True, info=None, bboxes=None, mask_image=None, history=None, **kwargs):
        '''
            action: Position, Quaternion (xyzw) of the goal
        '''
        if execute:
            assert env is not None, "Environment is required to execute the skill"
            floor_num = kwargs['floor_num']
            bld = kwargs['bld']
            pid = change_map(floor_num=floor_num, bld=bld, empty=True)
            move_behind_flag = U.confirm_user(execute, 'Do you want to continue (y/n)?', info_string='The robot will move behinding by like 1 meter.')
            if move_behind_flag:
                self.move_behind(env)
                bboxes = None
                mask_image = None
            env.reset(reset_arms=True, reset_pose=RP.HOME_L_PUSH_R_H)
            obs_pp = VU.get_obs(env, self.tf_base)
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
        cur_pos, cur_ori = None, None
        if env is not None:
            cur_pos = env.tiago.base.odom_listener.get_most_recent_msg()['pose'][:3]
            cur_ori = env.tiago.base.odom_listener.get_most_recent_msg()['pose'][3:]
        text_direction = ''
        info_cp = copy.deepcopy(info)
        prompt_rgb = rgb.copy()
        img_size = min(rgb.shape[0], rgb.shape[1])

        if self.oracle_action:
            # ask the user for direction
            direction = 'right'
            response = ''
            text_direction = 'left'
            return_info = {
                'response': response,
                'model_out': text_direction,
                'error_list': [],
            }
            gsam_query = ['all objects']
            if (bboxes is None) or (mask_image is None):
                bboxes, mask_image = self.get_object_bboxes(rgb, query=gsam_query)
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
            radius_per_pixel = self.prompt_args['radius_per_pixel']
            prompt_args.update({
                'radius': int(img_size * radius_per_pixel),
                'fontsize': int(img_size * 30 * radius_per_pixel),
            })
            prompt_rgb, obj_bbox_list = bbox_prompt_img(
                im=rgb.copy(),
                info=info_cp,
                bboxes=bboxes,
                prompt_args=prompt_args,
            )
            prompt_rgb_obj = prompt_rgb.copy()
            print(colored(f"Remember the object id you want to select. The object id is the character marked in circle next to the object.", 'yellow'))
            plt.imshow(prompt_rgb)
            plt.show()
            object_id = input("Enter the object id: ")
            bbox_selected = [bbox for bbox in obj_bbox_list if bbox.obj_id.lower() == object_id.lower()]
            obj_bbox_selected = [bbox.bbox for bbox in bbox_selected]
            env_id = obj_bbox_selected[0][0]
            obj_pos_f = self.extract_pcd_from_env_id(
                pcd=pcd,
                env_id=env_id,
                filter_nan=True,
                mask_image=np.asarray(mask_image),
            )
            # remove obj_pos_f that have z value less than 0.01
            obj_pos_high = obj_pos_f[obj_pos_f[:,2] > 0.01]
            obj_pos = np.mean(obj_pos_f, axis=0) # this is plot_start point for the arrow
            # take average of the min and max y position of the obj_pos_f
            obj_pos[1] = np.mean([np.min(obj_pos_high[:,1]), np.max(obj_pos_high[:,1])])
            text_directions = ['forward', 'left', 'right']
            print(colored(f"Which direction do you want to push the object? Choose one of the following directions: {text_directions} (F, L, R)", 'yellow'))
            text_direction = input("Enter the direction (F, L, R): ")
            if text_direction.lower() in ['r', 'l', 'f']:
                if text_direction.lower() == 'r':
                    text_direction = 'right'
                elif text_direction.lower() == 'l':
                    text_direction = 'left'
                elif text_direction.lower() == 'f':
                    text_direction = 'forward'
            return_info['object_id'] = object_id
            return_info['text_direction'] = text_direction
            return_info['model_out'] = [object_id, text_direction]
            return_info['model_response'] = ['', '']
            return_info['error_list'] = []
            return_info['model_analysis'] = ''
        if run_vlm:
            if self.add_histories:
                if history is None:
                    history = []
                history.extend(self.history_list)
            history_msgs = None
            if (history is not None) and (len(history)>0):
                history_msgs = self.create_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                    image_key='image_obj'
                )
            cur_pos_base = np.asarray([0.0,0.0,0.0])
            cur_ori_base = np.asarray([0.0,0.0,0.0,1.0])
            info_cp['save_key'] = info['save_key'] + '_obj'

            ################# query for which object to move
            gsam_query = ['all objects']
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
            U.save_image(overlay_image, os.path.join(self.vis_dir, f'overlay_image_{info_cp["save_key"]}.png'))
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
            radius_per_pixel = self.prompt_args['radius_per_pixel']
            prompt_args.update({
                'radius': int(img_size * radius_per_pixel),
                'fontsize': int(img_size * 30 * radius_per_pixel),
            })
            prompt_rgb, obj_bbox_list = bbox_prompt_img(
                im=rgb.copy(),
                info=info_cp,
                bboxes=bboxes,
                prompt_args=prompt_args,
            )
            prompt_rgb_obj = prompt_rgb.copy()
            info_cp['obj_bbox_list'] = obj_bbox_list
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_img_{info_cp["save_key"]}.png'))
            encoded_image = U.encode_image(prompt_rgb)
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=None,
                make_prompt_func=make_prompt_object,
                make_prompt_func_kwargs={
                    'query': query,
                    'info': info_cp,
                }
            )
            U.save_image(depth.astype(np.uint8), os.path.join(self.vis_dir, f'depth_{info_cp["save_key"]}.png'))
            obj_bbox_selected, object_id, return_info = self.get_bbox_from_response(response, obj_bbox_list, info_cp)
            # add the env_id to the obj_bbox_selected.
            self.save_model_output(
                rgb=prompt_rgb,
                response=response,
                subtitles=[f'Task Query: {query}', f'Object Id: {object_id}'],
                img_file=os.path.join(self.vis_dir, f'output_{info_cp["save_key"]}.png'),
            )
            if obj_bbox_selected is None:
                error = "Object id not found in the scene. Please select the object id from the image."
                return self.on_failure(
                    reason_for_failure=error,
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )

            # ################# query for which direction to move
            obj_bbox_selected = [obj_bbox_selected[0].bbox]
            prompt_rgb = rgb.copy()
            prompt_rgb, obj_bbox_list = bbox_prompt_img(
                im=rgb.copy(),
                info=info_cp,
                bboxes=obj_bbox_selected,
                prompt_args=prompt_args,
            )
            env_id = obj_bbox_selected[0][0]
            obj_pos_f = self.extract_pcd_from_env_id(
                pcd=pcd,
                env_id=env_id,
                filter_nan=True,
                mask_image=np.asarray(mask_image),
            )
            if len(obj_pos_f) == 0:
                error = "Depth value for the object is nan. Adjust the base position of the robot to get better depth values."
                return self.on_failure(
                    reason_for_failure=error,
                    reset_required=False,
                    capture_history={},
                    return_info={},
                )
            obj_pos_high = obj_pos_f[obj_pos_f[:,2] > 0.01]
            obj_pos = np.mean(obj_pos_f, axis=0) # this is plot_start point for the arrow
            if len(obj_pos_high) > 0:
                obj_pos[1] = np.mean([np.min(obj_pos_high[:,1]), np.max(obj_pos_high[:,1])])
            text_direction = 'forward'
            info_cp.update({
                'save_key': info['save_key'] + '_dir',
                'plot_robot_pos': obj_pos,
                'plot_robot_ori': cur_ori_base,
            })
            obj_px_pos = ((obj_bbox_selected[0][1] + obj_bbox_selected[0][3]) // 2, (obj_bbox_selected[0][2] + obj_bbox_selected[0][4]) // 2)
            print(f"Object position: {obj_pos}, Object pixel position: {obj_px_pos}")
            prompt_args.update({
                'start_point': obj_px_pos,
                'arrow_length': int(self.prompt_args['arrow_length_per_pixel'] * img_size),
            })
            if self.prompt_args['plot_direction']:
                prompt_rgb = prompt_move_img(
                    im=prompt_rgb,
                    prompt_args=prompt_args,
                    info=info_cp,
                )
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_{info_cp["save_key"]}.png'))

            encoded_image = U.encode_image(prompt_rgb)
            if (history is not None) and (len(history)>0):
                history_msgs = self.create_history_msgs(
                    history,
                    func=make_history_prompt_dir,
                    func_kwargs={},
                    image_key='image'
                )
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=history_msgs,
                make_prompt_func=make_prompt_dir,
                make_prompt_func_kwargs={
                    'query': query,
                    'info': info_cp,
                }
            )
            text_direction, _return_info = self.get_dir_from_response(response, info_cp)
            return_info['error_list'].extend(_return_info['error_list'])
            return_info['response'] = [return_info['response'], response]
            return_info['model_out'] = [return_info['model_out'], text_direction]
            self.save_model_output(
                rgb=prompt_rgb,
                response=response,
                subtitles=[f'Task Query: {query}', f'Direction: {text_direction}'],
                img_file=os.path.join(self.vis_dir, f'output_{info_cp["save_key"]}.png'),
            )

        capture_history = {
            'image_obj': prompt_rgb_obj,
            'image': prompt_rgb,
            'query': query,
            'model_response': [object_id, text_direction],
            'full_response': response,
            'text_direction': text_direction,
            'object_id': object_id,
            'model_analysis': '',
        }

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

        # base_footprint to odom transform
        approach_pos1_odom, approach_pos2_odom, approach_pos3_odom, goto_pos_odom = None, None, None, None
        approach_pos2_map, approach_ori2_map = None, None
        y_pos = obj_pos[1]
        x_pos = obj_pos[0] + 0.15 # average x position of the object + 30 cm forward
        align_pos_base = np.array([0.0, obj_pos[1], 0.0]) # first align the robot to the object base
        distance_before_push_f = 0.65
        distance_before_push_s = 0.50

        # first move the robot to the right most position for left push and vice versa.
        ext_right_pos_y = min(obj_pos_f[:, 1])
        ext_left_pos_y = max(obj_pos_f[:, 1])
        distance4hand_y = 0.20
        distance4hand_x = 0.50
        arm_joint_angle_series = []
        push_pos, push_ori = None, None

        if text_direction == 'forward':
            align_pos_base[1] += 0.1
            approach_pos1 = np.array([max(np.min(obj_pos_f[:,0]) - distance_before_push_f, 0.0), y_pos+0.1, 0.0]) # be 30 cm away from the object
            approach_pos2 = copy.deepcopy(approach_pos1)
            approach_ori2 = np.asarray([0.0, 0.0, 0.0, 1.0])
            push_ori = np.asarray([0.0, 0.0, 0.0, 1.0])
            push_pos = copy.deepcopy(approach_pos1)
            push_pos[0] += 0.70
            goto_ori = np.asarray([0.0, 0.0, 0.0, 1.0])
            goto_pos = np.asarray([0.0, 0.0, 0.0])
        elif text_direction == 'right':
            ext_x_pos = np.min(obj_pos_f[:, 0])
            align_pos_base = np.array([ext_x_pos - distance4hand_x, ext_left_pos_y + distance4hand_y, 0.0])
            approach_pos1 = align_pos_base
            approach_pos2 = align_pos_base
            approach_ori2 = np.asarray([0.0, 0.0, 0.0, 1.0])
            arm_joint_angle_series = [
                [0.124, 1.423, 1.805, 1.778, -1.329, 1.392, -0.008],
                [1.39, 0.80, 0.346, 1.293, -1.186, 0.974, -0.061],
                [1.50, 0.051, 0.085, 0.240, -1.389, -0.271, -0.208],
                [0.0, 0.051, 0.085, 0.240, -1.389, -0.271, -0.208],
                [-1.107, 1.466, 2.703, 1.718, -1.414, 1.391, 0.002],
            ]
            goto_pos = np.array([x_pos, y_pos, 0.0])
            goto_ori = np.asarray([0.0, 0.0, 0.0, 1.0])
        elif text_direction == 'left':
            ext_x_pos = np.min(obj_pos_f[:,0])
            align_pos_base = np.array([ext_x_pos - distance4hand_x, ext_right_pos_y - distance4hand_y, 0.0])
            approach_pos1 = align_pos_base
            approach_pos2 = align_pos_base
            approach_ori2 = np.asarray([0.0, 0.0, 0.0, 1.0])
            arm_joint_angle_series = [
                [0.124, 1.423, 1.805, 1.778, -1.329, 1.392, -0.008],
                [1.39, 0.80, 0.346, 1.293, -1.186, 0.974, -0.061],
                [1.50, 0.051, 0.085, 0.240, -1.389, -0.271, -0.208],
                [0.0, 0.051, 0.085, 0.240, -1.389, -0.271, -0.208],
                [-1.107, 1.466, 2.703, 1.718, -1.414, 1.391, 0.002],
            ]
            goto_pos = np.array([x_pos, y_pos, 0.0])
            goto_ori = np.asarray([0.0, 0.0, 0.0, 1.0])
        else: raise NotImplementedError(f"Direction {text_direction} not implemented")

        align_pos_odom, approach_pos1_odom, approach_pos2_odom, approach_pos3_odom, goto_pos_odom = None, None, None, None, None
        if env is not None:
            transform = T.pose2mat((self.tf_odom.get_transform(target_link='/base_footprint')))
            transform_map = T.pose2mat((self.tf_map.get_transform(target_link='/base_footprint')))

            align_pos_odom = transform @ np.concatenate((align_pos_base, [1.0]))
            align_pos_odom = align_pos_odom[:3]

            approach_pos1_odom = transform @ np.concatenate((approach_pos1, [1.0]))
            approach_pos1_odom = approach_pos1_odom[:3]

            approach_pos2_odom = transform @ np.concatenate((approach_pos2, [1.0]))
            approach_pos2_odom = approach_pos2_odom[:3]

            approach_pose_base = T.pose2mat((approach_pos2, approach_ori2))
            approach_pose2_map = transform_map @ approach_pose_base
            approach_pos2_map = approach_pose2_map[:3, 3]
            approach_ori2_map = T.mat2quat(approach_pose2_map[:3, :3])

            push_pos_odom, push_ori_odom = None, None
            if push_pos is not None:
                push_pose_base = T.pose2mat((push_pos, push_ori))
                push_pose_odom = transform @ push_pose_base
                push_pos_odom = push_pose_odom[:3, 3]
                push_ori_odom = T.mat2quat(push_pose_odom[:3, :3])

            goto_pos_odom = transform @ np.concatenate((goto_pos, [1.0]))
            goto_pos_odom = goto_pos_odom[:3]
            goto_pose_base = T.pose2mat((goto_pos, goto_ori))
            goto_pose_map = transform_map @ goto_pose_base
            goto_pos_map = goto_pose_map[:3, 3]
            goto_ori_map = T.mat2quat(goto_pose_map[:3, :3])

        if self.debug:
            pcd_wrt_odom = np.concatenate((pcd.reshape(-1, 3), np.ones((pcd.reshape(-1,3).shape[0], 1))), axis=1)
            transform = T.pose2mat((self.tf_odom.get_transform(target_link=f'/base_footprint')))
            pcd_wrt_odom = (transform @ pcd_wrt_odom.T).T
            pcd_wrt_odom = pcd_wrt_odom[:, :3]
            pcd_to_plot = pcd_wrt_odom.reshape(-1,3)
            rgb_to_plot = rgb.reshape(-1,3)

            # add the align_pos_odom to the pcd_to_plot in color yellow
            pcd_to_plot = np.concatenate((pcd_to_plot, align_pos_odom.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[255.0, 255.0, 0.0]])), axis=0) # align pos in yellow
            # add the approach_pos1_odom to the pcd_to_plot in color red
            pcd_to_plot = np.concatenate((pcd_to_plot, approach_pos1_odom.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[255.0, 0.0, 0.0]])), axis=0) # approach pos in red
            # add the approach_pos2_odom to the pcd_to_plot in color green
            pcd_to_plot = np.concatenate((pcd_to_plot, approach_pos2_odom.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[0.0, 255.0, 0.0]])), axis=0) # approach pos in green

            if push_pos_odom is not None:
                pcd_to_plot = np.concatenate((pcd_to_plot, push_pos_odom.reshape(1,3)), axis=0)
                rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[0.0, 255.0, 0.0]])), axis=0) # push pos in green

            pcd_to_plot = np.concatenate((pcd_to_plot, goto_pos_odom.reshape(1,3)), axis=0)
            rgb_to_plot = np.concatenate((rgb_to_plot.reshape(-1,3), np.asarray([[0.0, 0.0, 255.0]])), axis=0) # goal pos in blue

            U.plotly_draw_3d_pcd(pcd_to_plot, rgb_to_plot)

        is_success = True
        duration_scale_factor = 1.2

        execute = U.confirm_user(execute, "Do you want to continue? (y/n): ", f"Using the remove obstacle skill for {text_direction} direction")
        if execute:
            if text_direction == 'forward':
                print(colored("putting left arm in home and right arm in push pose", "red"))
                print(colored("do not say NO here unless unsafe", "red"))
                grasp_h_r = RP.HOME_L_PUSH_R
                _exec = U.reset_env(env, reset_pose=grasp_h_r, reset_pose_name='HOME_L_PUSH_R', delay_scale_factor=1.5)


            # align the robot to the object
            arm = 'right' if ((text_direction == 'forward') or (text_direction == 'right')) else 'left'
            if text_direction == 'forward':
                self.tuck_in_gripper(env, arm)
                env.reset(reset_arms=False, reset_pose={'left': None, 'right': None, 'torso': 0.05}, allowed_delay_scale=6.0, wait_user=True) # go down the torso

            success = self.goto_odom_pos(env, align_pos_odom)
            rospy.sleep(0.5)
            success = self.goto_odom_pos(env, approach_pos1_odom)
            rospy.sleep(0.5)
            success = self.goto_odom_pos(env, approach_pos2_odom)
            rospy.sleep(0.5)
            print("Continue to rotating in place?")
            input("Press any key to continue")

            print(approach_pos2_map, approach_ori2_map)
            goal = self.create_move_base_goal((approach_pos2_map, approach_ori2_map)) # this should only rotate the robot in place
            state = self.send_move_base_goal(goal)
            if text_direction == 'forward':
                env.tiago.torso.torso_writer.write(env.tiago.torso.create_torso_command(0.05))
            else:
                env.tiago.torso.torso_writer.write(env.tiago.torso.create_torso_command(0.10))

            if text_direction == 'forward':
                print(colored("The hand will tuck out and push the object quickly. Only press yes if there is enough space to tuck out gripper.", "red"))
                U.clear_input_buffer()
                input("Press Enter to continue...")
                self.close_gripper(env=env, side=arm)
                self.tuck_out_gripper(env, arm)
                self.push_pull_arm(env, arm, goto_pos=push_pos_odom)
                self.open_gripper(env=env, side=arm)
                ## Reseting arms to home position before moving back
                reset_dict = RP.HOME_L_HOME_R
                reset_dict['torso'] = None
                env.reset(reset_arms=True, reset_pose=reset_dict)
            else:
                for joint_angles in arm_joint_angle_series:
                    cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
                    duration_scale = 2*duration_scale_factor*np.linalg.norm(joint_angles-cur_joint_angles) # move twice as low
                    print("duration scale: ", duration_scale)
                    env.tiago.arms[arm].write(joint_angles, duration_scale, threshold=0.01, delay_scale_factor=duration_scale_factor)
            goal = self.create_move_base_goal((goto_pos_map, goto_ori_map))
            state = self.send_move_base_goal(goal)

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
