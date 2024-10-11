import os
import sys
import copy
import numpy as np
import pickle

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

import bumble
from bumble.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from bumble.tiago.skills.base import SkillBase, movebase_code2error
import bumble.utils.utils as U
import bumble.utils.transform_utils as T # transform_utils
from bumble.tiago.prompters.direction import prompt_move_img, prompt_rotate_img
from bumble.tiago.prompters.object_bbox import bbox_prompt_img

from termcolor import colored

def make_prompt(query, info, llm_baseline_info=None, method="ours", histories=None, **kwargs):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    add_dist_info = False
    bbox_ind2dist = None
    if info['add_dist_info'] == True:
        add_dist_info = True
        obj_bbox_list = info['obj_bbox_list']
        bbox_id2dist = info['bbox_id2dist']
        bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]
    # bbox_id2dir = info['bbox_id2dir']
    if method == "ours":
        visual_instructions = [
            "the image",
            "the direction of the image",
            "The forward direction is moving in the direction of the image (towards the top of the image), and backward is moving in the opposite direction. The left direction is moving to the left of the image, and right is moving to the right of the image. The robot uses its left arm to grab objects and can easily grasp objects on the left side of the image. If the robot moves right, the object will move to the left side of the image. If the robot moves left, the objects will move to the right side of the image. If the robot moves forward, the objects in the front will be closer and move to bottom of the image. If the robot moves backward, the objects in the front will move farther away from the robot, towards the top of the image. Each object is marked with an object id, example, 'B'. Along with it, the image is marked with directions indicating left, forward, and right directions to help you decide the direction.",
            "describe the scene and each object id. Then,",
            "Make use of the markers (F,L,R) to guide your answer. ",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a description",
            "forward",
            "The forward direction is moving toward the objects on the scene, and backward is moving away from the objects on the scene. The left direction is moving to the left of the scene, and right is moving to the right of the scene. The robot uses its left arm to grab objects and can easily grasp objects on the left side of the scene. If the robot moves right, the object will move to the left side of the scene. If the robot moves left, the objects will move to the right side of the scene. If the robot moves forward, the objects in the front will be closer. If the robot moves backward, the objects in the front of the scene will move farther away from the robot.",
            "",
            "",
        ]
    elif method == "ours_no_markers":
        visual_instructions = [
            "the image",
            "the direction of the image",
            "The forward direction is moving in the direction of the image (towards the top of the image), and backward is moving in the opposite direction. The left direction is moving to the left of the image, and right is moving to the right of the image. The robot uses its left arm to grab objects and can easily grasp objects on the left side of the image. If the robot moves right, the object will move to the left side of the image. If the robot moves left, the objects will move to the right side of the image. If the robot moves forward, the objects in the front will be closer and move to bottom of the image. If the robot moves backward, the objects in the front will move farther away from the robot, towards the top of the image.",
            "describe the scene. Then,",
            "",
        ]
    else:
        raise NotImplementedError

    instructions = f"""
INSTRUCTIONS:
You are tasked to predict the direction in which the robot must move to complete the task. You are provided with {visual_instructions[0]} of the scene, and a description of the task. The robot is currently facing {visual_instructions[1]}. The robot can move in ONLY ONE of the four directions: forward, backward, left, or right by a distance of {info['move_dist']} meters. {visual_instructions[2]}

You are a five-time world champion in this game. Output only one of the directions: forward, backward, left, or right. Do NOT leave it empty. First, summarize all the errors made in previous predictions if provided. Then, {visual_instructions[3]}describe the effect of the robot moving in each direction. {visual_instructions[4]}Then, select the direction that can best help complete the task of reaching near the object of interest. Finally, provide the direction in a valid JSON of this format:
{{"direction_to_move": ""}}
""".strip()
    if add_dist_info:
        instructions += f"""\n
Below is provided the distances to the objects in the scene. Use this information to decide how far the robot is from the desired object."""
        for obj_id, dist in bbox_ind2dist:
            instructions += f"""
- Object id {obj_id} is {dist:.2f} metres from the robot."""
        instructions += f"""\n"""

    if llm_baseline_info:
        instructions += f"""\n
SCENE DESCRIPTION:
{llm_baseline_info['im_scene_desc']}
OBJECT ID DESCRIPTIONS:
{llm_baseline_info['obj_descs']}
"""
    task_prompt = f"""\nTASK DESCRIPTION: {query}"""
    task_prompt += f"""\n
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

def make_history_prompt(history, _type='failure'):
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
- Answer: {msg['text_direction']}
""".strip()
        history_desc.append(example_desc)
        history_model_analysis.append(msg['model_analysis'])
    return instructions, history_desc, history_model_analysis

class MoveToSkill(SkillBase):
    def __init__(
            self,
            oracle_action=False,
            debug=False,
            run_dir=None,
            prompt_args=None,
            skip_ros=False,
            add_histories=False,
            *args, **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.move_length = 0.3 # 30 cm
        self.oracle_action = oracle_action
        self.debug = debug
        self.skip_ros = skip_ros
        self.add_histories = add_histories
        self.history_list = None
        if self.add_histories:
            base_dir = os.path.join(bumble.__path__[0], 'long_term_mem', 'move_base')
            history_eval_dirs = [os.path.join(base_dir, 'eval_id001.pkl'), os.path.join(base_dir, 'eval_id000.pkl'), os.path.join(base_dir, 'eval_id003.pkl')]
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

        if not self.skip_ros:
            self.setup_listeners()
        self.vis_dir = os.path.join(run_dir, 'move_to')
        os.makedirs(self.vis_dir, exist_ok=True)
        self.add_obj_id = prompt_args.get('add_obj_id', True)
        arrow_length_per_pixel = prompt_args.get('arrow_length_per_pixel', 0.15)
        radius_per_pixel = prompt_args.get('radius_per_pixel', 0.04)
        if self.method != "ours":
            plot_direction = False
        else:
            plot_direction = prompt_args.get('plot_direction', True)
        self.add_dist_info = prompt_args.get('add_dist_info', True)
        self.prompt_args = {
            "add_arrows": prompt_args.get('add_arrows', True),
            "color": (0, 0, 0),
            "mix_alpha": 0.6,
            'thickness': 2,
            'rgb_scale': 255,
            'plot_dist_factor': prompt_args.get('plot_dist_factor', 1.0),
            'add_dist_info': self.add_dist_info,
            'rotate_dist': 0.3,
            'radius_per_pixel': radius_per_pixel,
            'arrow_length_per_pixel': arrow_length_per_pixel,
            'add_object_boundary': prompt_args.get('add_object_boundary', False),
            'plot_direction': plot_direction,
        }
        self.skill_name = "move"
        self.skill_descs = f"""
skill_name: move
arguments: direction
description: Moves the robot base in the specified direction by {self.move_length} meters. The direction can be either 'forward', 'backward', 'left', 'right' w.r.t. the camera view. This skill should only be used to adjust the base of the robot within few meters and not for long distance navigation.
""".strip()

    def get_param_from_response(self, response, info):
        return_info = {}
        return_info['response'] = response
        direction = None
        error_list = []
        try:
            direction = U.extract_json(response, 'direction_to_move')
            print(f"Direction: {direction}")
            if direction.lower() not in ['forward', 'backward', 'left', 'right']:
                error = 'Invalid direction. Please provide one of the following directions: forward, backward, left, right.'
                error_list.append(error)
                direction = None
        except Exception as e:
            print(f"Error: {e}")
            error = 'Invalid response format. Please provide the direction in a valid JSON format.'
            error_list.append(error)
            direction = None
        return_info['model_out'] = direction
        return_info['error_list'] = error_list
        return direction, return_info

    def step(self, env, rgb, depth, pcd, normals, query, execute=True, run_vlm=True, info=None, history=None, bboxes=None, mask_image=None, **kwargs):
        '''
            action: Position, Quaternion (xyzw) of the goal
        '''
        print("MoveToSkill: Move to the initial position")
        # get the current position of the robot_base w.r.t. odom
        if execute:
            assert env is not None, "Environment is required to execute the skill"
        cur_pos, cur_ori = None, None
        if env is not None:
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
            if self.add_histories:
                if history is None:
                    history = []
                history.extend(self.history_list)
            info_cp = copy.deepcopy(info)
            prompt_rgb = None
            prompt_rgb = rgb.copy()
            img_size = min(rgb.shape[0], rgb.shape[1])
            if self.add_obj_id: # add object ids to the image
                gsam_query = ['all objects']
                if (bboxes is None) or (mask_image is None):
                    bboxes, mask_image = self.get_object_bboxes(rgb, query=gsam_query)
                if len(bboxes) == 0:
                    # this should not happen
                    import ipdb; ipdb.set_trace()
                    error = "No objects found in the scene."
                    self.on_failure(
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
                info_cp.update({'add_dist_info': self.add_dist_info})

                prompt_rgb, obj_bbox_list = bbox_prompt_img(
                    im=rgb.copy(),
                    info=info_cp,
                    bboxes=bboxes,
                    prompt_args=prompt_args,
                )
                info_cp['obj_bbox_list'] = obj_bbox_list
                if self.method == "ours_no_markers":
                    prompt_rgb = rgb.copy()
                U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_img_{info["save_key"]}.png'))


            # w.r.t. the base_footprint frame
            info_cp['plot_robot_pos'] = np.asarray([0.0,0.0,0.0]) # cur_pos_base
            info_cp['plot_robot_ori'] = np.asarray([0.0,0.0,0.0,1.0]) # cur_ori_base
            info_cp['skill_type'] = 'move'
            info_cp['move_dist'] = self.move_length # in meters
            self.prompt_args.update({
                'radius': int(img_size * self.prompt_args['radius_per_pixel']),
                'fontsize': int(img_size * 30 * self.prompt_args['radius_per_pixel']),
                'start_point': (rgb.shape[1]//2, rgb.shape[0] - 20),
                'arrow_length': int(self.prompt_args['arrow_length_per_pixel'] * img_size),
            })

            if self.prompt_args['plot_direction']:
                prompt_rgb = prompt_move_img(
                    im=prompt_rgb,
                    prompt_args=self.prompt_args,
                    info=info_cp,
                )
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_{info["save_key"]}.png'))
            # get the direction from the model
            encoded_image = U.encode_image(prompt_rgb)
            history_msgs = None
            if (history is not None) and (len(history)>0):
                history_msgs = self.create_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                )
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=history_msgs,
                make_prompt_func=make_prompt,
                make_prompt_func_kwargs={
                    'query': query,
                    'info': info_cp,
                },
            )
            text_direction, return_info = self.get_param_from_response(response, info)
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

        if text_direction.lower() == 'forward':
            pos = [self.move_length, 0.0, 0.0]
        elif text_direction.lower() == 'backward':
            pos = [-self.move_length, 0.0, 0.0]
        elif text_direction.lower() == 'left':
            pos = [0.0, self.move_length, 0.0]
        elif text_direction.lower() == 'right':
            pos = [0.0, -self.move_length, 0.0]
        else:
            print("Invalid direction")
            print("This error should be captured in get_param_from_response")
            raise ValueError("Invalid direction")

        if not self.skip_ros:
            transform = T.pose2mat((self.tf_map.get_transform(target_link=f'/base_footprint')))
            pose_map = T.pose2mat((pos, [0.0, 0.0, 0.0, 1.0]))
            pose_map = transform @ pose_map
            pos_map = pose_map[:3, 3]
            ori_map = T.mat2quat(pose_map[:3, :3])
            print(f"Calculated Position in map: {pos_map}")
            print(f"Calculated Orientation in map: {ori_map}")

            goal_pos_map = pos_map
            goal_ori_map = ori_map # goal in map frame
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
        execute = U.confirm_user(execute, "Do you want to continue? (y/n): ", f"Using the move skill for {text_direction} direction")
        if execute:
            goal = self.create_move_base_goal((goal_pos_map, goal_ori_map))
            state = self.send_move_base_goal(goal)
            if state == GoalStatus.SUCCEEDED:
                is_success = True
            else:
                error = movebase_code2error(state)
                is_success = False

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
