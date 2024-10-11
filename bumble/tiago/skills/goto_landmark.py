import os
import atexit
import cv2
import sys
import copy
import numpy as np

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from actionlib_msgs.msg import GoalStatus

from bumble.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from bumble.tiago.skills.base import SkillBase, movebase_code2error
from bumble.tiago.prompters.direction import add_circle_to_image
import bumble.utils.utils as U
import bumble.tiago.RESET_POSES as RP
import bumble.utils.transform_utils as T # transform_utils
from bumble.tiago.ros_restrict import change_map

from termcolor import colored

# some of the locations are fake since the landmarks are not important to solve the evaluated tasks.
landmark2poses_floor1 = {
    'ahg1_1': ((-2.0, -12.39, 0.0), (0.0, 0.0, 0.013277470784416425, 0.9999118504996173)),
    'ahg1_2': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'ahg1_3': ((-31.87, -13.42, 0.0), (0.0, 0.0, 0.9993860627871034,  0.03503566050314777)),
    'ahg1_4': ((-33.36, 38.72, 0.0), (0.0, 0.0, 0.6602350956429588, 0.7510589980030418)),
    'ahg1_5': ((-28.97, 40.23, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'ahg1_6': ((-31.87, -13.42, 0.0), (0.0, 0.0, 0.9993860627871034,  0.03503566050314777)),
}
landmark2poses_floor2 = {
    'ahg2_1': ((-31.24, -8.71, 0.0), (0.0, 0.0, -0.69, 0.72)),
    'ahg2_2': ((-3.0, -8.7, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'ahg2_3': ((-15.93, -9.18, 0.0), (0.0, 0.0, 0.74, 0.67)),
    'ahg2_4': ((-38.92, -9.36, 0.0), (0.0, 0.0, 0.7409200876041254, 0.6715931981376041)),
}

mbb_landmark2poses_floor1 = {
    # biorad.jpg  elevator.jpg  exit.jpg  test.jpg  trash.jpg
    'mbb1_1': ((-2.045, -0.96, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'mbb1_2': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'mbb1_3': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'mbb1_4': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'mbb1_5': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
}
mbb_landmark2poses_floor2 = {
    'mbb2_1': ((-5.26, 2.36, 0.0), (0.0, 0.0, 0.7169423564956282, 0.6971324533132105)),
    'mbb2_2': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)), # random numbers
    'mbb2_3': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)), # random numbers
    'mbb2_4': ((-6.257545365994412, -6.837596546561547, 0.0), (0.0, 0.0, -0.6893903334285728, 0.7243900663145797)),
    'mbb2_5': ((0.373, 2.496, 0.0), (0.0, 0.0, -1.0, 0.0)),
}
mbb_landmark2poses_floor3 = {
    'mbb3_1': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'mbb3_2': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'mbb3_3': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
}

nhb_landmark2poses_floor3 = {
    'nhb3_1': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'nhb3_2': ((5.708, -5.02, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'nhb3_3': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'nhb3_4': ((0.516, 3.829, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'nhb3_5': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
}

# cool_lab.jpg  dining_area.jpg  kitchen.jpg  lab_coat.jpg  no_entry_door.jpg  restroom.jpg  stairs.jpg
nhb_landmark2poses_floor4 = {
    'nhb4_1': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'nhb4_2': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'nhb4_3': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'nhb4_4': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
    'nhb4_5': ((-23.0, 39.29, 0.0), (0.0, 0.0, 0.0, 1.0)),
}

floors = {
    'ahg': {
        1: landmark2poses_floor1,
        2: landmark2poses_floor2,
    },
    'mbb': {
        1: mbb_landmark2poses_floor1,
        2: mbb_landmark2poses_floor2,
        3: mbb_landmark2poses_floor3,
    },
    'nhb': {
        3: nhb_landmark2poses_floor3,
        4: nhb_landmark2poses_floor4,
    }
}

def make_prompt(query, info, llm_baseline_info=None, method="ours"):
    '''
        The below instructions are used to prompt the model for skill parameter prediction and not skill selection.
        query: str is the main (sub)task description
        info: dictionary of information required to prompt the model
    '''
    if method == "ours":
        visual_instructions = [
            "multiple images concatenated together",
            "Each image represents a landmark, example, bedroom, on the current floor. Each image is marked with a landmark ID, example, 'B'. ",
            " each of the scenes in the image. Then, describe",
        ]
    elif method == "llm_baseline":
        visual_instructions = [
            "a list of descriptions of landmarks with their ID, example, 'B'",
            "",
            "",
        ]
    else:
        raise NotImplementedError
    instructions = f"""
INSTRUCTIONS:
You are tasked to select the landmark where the robot must navigate to best complete or make progress towards the task. You are provided with {visual_instructions[0]} along with the task description, and a brief summary of landmarks in each floor. {visual_instructions[1]}You are required to select the landmark ID that the robot must navigate to best complete or make progress towards completing the task. Each floor has different landmarks. If you do not find the landmark described in the task, you can output an error and go to the elevator by selecting the corresponding landmark id. The error can be of the form: "Room that looks like classroom is present in floor 3 and 5, but the robot is currently in floor 2." If choosing elevator, always provide the floor the robot should go to in the error. If the landmark is present in the current floor, you can keep error as empty: ""

You are a five-time world champion in this game. Output only one landmark ID. Do NOT leave it empty. First, describe{visual_instructions[2]} what are the kind of objects you will find in this room. Then, give an analysis of how you would chose the landmark to best complete the task. If you do not see a landmark where you can potentially find the object, list down one landmarks in other floors where you can find the object, and go to the landmark corresponding to the elevator of this floor. Then, select the landmark ID that can best help complete the task. Finally, provide the landmark ID in a valid JSON of this format:
{{"landmark_id": "", "error": ""}}

SUMMARY OF LANDMARKS:
{info['summary']}
""".strip()

    if llm_baseline_info:
        instructions += f"""\n
You are currently on a floor with these landmarks, described by ID:
{llm_baseline_info['obj_descs']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {query}"""
    task_prompt += f"""\n
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

def make_prompt_floor(info, llm_baseline_info=None, method="ours"):
    instructions = f"""
INSTRUCTIONS:
You are tasked to briefly summarize the landmarks. You are provided with multiple images concatenated together. Each image represents a landmark, example, bedroom. Each image is marked with a landmark ID, example, 'B'. You are required to provide a brief summary of the landmarks in a valid JSON format: {{"summary": ""}}

You are a five-time world champion in this game. In the summary, first, describe each of the scenes in the image marked by the landmark ID. Then, describe what are the kind of objects you will find in this room.
""".strip()
    task_prompt = f"""\n
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

class GoToLandmarkSkill(SkillBase):
    def __init__(
            self,
            bld,
            oracle_action=False,
            debug=False,
            run_dir=None,
            prompt_args=None,
            use_cache=True,
            *args, **kwargs,
        ):
        super().__init__(*args, **kwargs)
        # NOTE: We assume that the head is looking straight ahead
        self.oracle_action = oracle_action
        self.debug = debug
        self.use_cache = use_cache
        self.setup_listeners()
        if bld == 'ahg':
            self.landmark_keys = [\
                ['ahg1_1', 'ahg1_2', 'ahg1_3', 'ahg1_4', 'ahg1_5'], \
                ['ahg2_1', 'ahg2_2', 'ahg2_3'] \
            ]
        elif bld == 'mbb':
            self.landmark_keys = [ \
                ['mbb1_1', 'mbb1_2', 'mbb1_3', 'mbb1_4'], \
                ['mbb2_1', 'mbb2_2', 'mbb2_3', 'mbb2_4', 'mbb2_5'], \
                ['mbb3_1', 'mbb3_2', 'mbb3_3'], \
            ]
        elif bld == 'nhb':
            self.landmark_keys = [ \
                [], [], \
                ['nhb3_1', 'nhb3_2', 'nhb3_3', 'nhb3_4'], \
                ['nhb4_1', 'nhb4_2', 'nhb4_3', 'nhb4_4', 'nhb4_5'], \
            ]
        self.vis_dir = os.path.join(run_dir, 'goto_landmark')
        os.makedirs(self.vis_dir, exist_ok=True)
        radius_per_pixel = prompt_args.get('radius_per_pixel', 0.04)
        self.prompt_args = {
            'color': (0, 0, 0),
            'thickness': 2,
            'rgb_scale': 255,
            'radius_per_pixel': radius_per_pixel,
            'prompt_image_size': (256, 256),
            'mix_alpha': 0.6,
        }
        self.skill_name = "goto_landmark"
        self.skill_descs = f"""
skill_name: goto_landmark
arguments: Selected landmark image from the environment from various options.
description: Navigates to the landmark in the environment, example, bedroom, kitchen, tool shop, etc.
""".strip()
        self.brief_summary = self.prompt_all_floors(bld)

    def prompt_all_floors(self, bld):
        # check if there is a cache in the bumble/cache_landmark folder with bld_{bld}_landmark_images.txt
        # if yes, then load the cache and return the summary
        # if no, then create the cache and return the summary
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landmark_images', 'cache_landmark')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f'{bld}_landmark_images.txt')
        if self.use_cache and os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                summary_str = f.read()
            return summary_str

        img_size = 480
        self.prompt_args.update({
            'radius': int(img_size * self.prompt_args['radius_per_pixel']),
            'fontsize': int(img_size *  15 * self.prompt_args['radius_per_pixel']),
        })

        floor_nums = floors[bld].keys()
        summary_str = ''
        for floor_num in floor_nums:
            summary_str += f"\nFloor {floor_num}:\n"
            # start each ind with a character from A to Z
            # concatenate each image along the horizontal axis to create a prompt image
            prompt_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landmark_images', f'{bld}_landmark_images{floor_num}')
            landmark_keys = self.landmark_keys[floor_num-1]
            extension = '.png' if ((floor_num == 2) and (bld == 'ahg')) else '.jpg' # TODO: make image extension consistent
            rgb_images = [cv2.imread(os.path.join(prompt_img_dir, f'{landmark}{extension}')) for landmark in landmark_keys]
            rgb_images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in rgb_images]
            # # draw borders around the images
            rgb_images = [self.resize_image(im) for im in rgb_images]

            landmark_id2name = {}
            prompt_rgb = np.zeros((self.prompt_args['prompt_image_size'][0], len(landmark_keys)*self.prompt_args['prompt_image_size'][1], 3), dtype=np.uint8)
            for ind, (landmark, landmark_img) in enumerate(zip(landmark_keys, rgb_images)):
                label = chr(ord('A') + ind)
                landmark_id2name[label] = landmark
                plot_position = (landmark_img.shape[1]//2, 50)
                prompt_rgb_t = add_circle_to_image(
                    landmark_img.copy(),
                    [(plot_position, label)],
                    self.prompt_args,
                    info=None,
                )
                prompt_rgb[:, ind*self.prompt_args['prompt_image_size'][1]:(ind+1)*self.prompt_args['prompt_image_size'][1], :] = prompt_rgb_t

            encoded_image = U.encode_image(prompt_rgb)
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=None,
                make_prompt_func=make_prompt_floor,
                make_prompt_func_kwargs={
                    'info': None,
                },
                force_vlm_prompt=True,  # ignore baseline kwarg, we must pass in images to get room summary.
            )

            summary, return_info = self.get_param_from_response_summary(response)
            summary_str += f"{summary}\n"

        with open(cache_file, 'w') as f:
            f.write(summary_str)

        return summary_str

    def get_param_from_response_summary(self, response, info=None):
        return_info = {}
        error_list = []
        try:
            summary = U.extract_json(response, 'summary')
        except Exception as e:
            print(f"Error: {e}")
            error = 'Invalid response format. Please provide the landmark_id in a valid JSON format.'
            error_list.append(error)
            summary = None
        return_info['model_out'] = summary
        return_info['error_list'] = error_list
        return summary, return_info

    def get_param_from_response(self, response, info):
        return_info = {}
        landmark_desc = None
        landmark_id2name = info['landmark_id2name']
        return_info['response'] = response
        error_list = []
        try:
            landmark_id = U.extract_json(response, 'landmark_id')
            print(f"Landmark ID: {landmark_id}")
            if landmark_id.lower() not in [k.lower() for k in landmark_id2name.keys()]:
                error = 'Invalid landmark ID predicted. Please provide a valid landmark ID.'
                error_list.append(error)
                landmark_id = None
        except Exception as e:
            print(f"Error: {e}")
            error = 'Invalid response format. Please provide the landmark_id in a valid JSON format.'
            error_list.append(error)
            landmark_id = None

        try:
            error_to_pass = U.extract_json(response, 'error')
            print(f"Error for selector: {error_to_pass}")
        except Exception as e:
            print(f"Error: {e}")
            error = 'Invalid response format. Please provide the error in a valid JSON format.'
            error_list.append(error)
            error_to_pass = None

        return_info['model_out'] = [landmark_id, error_to_pass]
        return_info['error'] = error_to_pass
        return_info['error_list'] = error_list
        return_info['model_out'] = [landmark_id, error_to_pass, landmark_desc]
        return landmark_id, return_info

    def resize_image(self, img):
        # diff
        diff = abs(img.shape[0] - img.shape[1])
        # crop the bigger dimension to make it square from the center
        if img.shape[0] > img.shape[1]:
            img = img[diff//2:-diff//2, :]
        elif img.shape[0] < img.shape[1]:
            img = img[:, diff//2:-diff//2]
        img = cv2.resize(img, self.prompt_args['prompt_image_size'])
        return img

    def on_failure(self, floor_num, bld, *args, **kwargs):
        pid = change_map(floor_num=floor_num, bld=bld, empty=True)
        return super().on_failure(*args, **kwargs)

    def on_success(self, floor_num, bld, *args, **kwargs):
        pid = change_map(floor_num=floor_num, bld=bld, empty=True)
        return super().on_success(*args, **kwargs)

    def step(self, env, rgb, depth, pcd, normals, query, execute=True, run_vlm=True, info=None, **kwargs):
        '''
            action: Position, Quaternion (xyzw) of the goal
        '''
        assert 'floor_num' in kwargs.keys()
        assert 'bld' in kwargs.keys()
        floor_num = int(kwargs['floor_num'])
        bld = kwargs['bld']

        #### add map to restrict areas and publish zero map when function ends.
        pid = change_map(floor_num=floor_num, bld=bld, empty=True if bld == 'mbb' else False)
        #####

        prompt_rgb = rgb.copy()
        landmark, landmark_id = None, None
        response = ''

        if run_vlm:
            img_size = min(rgb.shape[0], rgb.shape[1])
            self.prompt_args.update({
                'radius': int(img_size * self.prompt_args['radius_per_pixel']),
                'fontsize': int(img_size *  15 * self.prompt_args['radius_per_pixel']),
            })

            # start each ind with a character from A to Z
            # concatenate each image along the horizontal axis to create a prompt image
            prompt_img_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'landmark_images', f'{bld}_landmark_images{floor_num}')
            landmark_keys = self.landmark_keys[floor_num-1]
            extension = '.png' if ((floor_num == 2) and (bld=='ahg')) else '.jpg' # change this later on.
            rgb_images = [cv2.imread(os.path.join(prompt_img_dir, f'{landmark}{extension}')) for landmark in landmark_keys]
            rgb_images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in rgb_images]
            # # draw borders around the images
            # rgb_images = [cv2.copyMakeBorder(im, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for im in rgb_images]
            rgb_images = [self.resize_image(im) for im in rgb_images]

            info_cp = copy.deepcopy(info)
            landmark_id2name = {}
            prompt_rgb = np.zeros((self.prompt_args['prompt_image_size'][0], len(landmark_keys)*self.prompt_args['prompt_image_size'][1], 3), dtype=np.uint8)
            for ind, (landmark, landmark_img) in enumerate(zip(landmark_keys, rgb_images)):
                label = chr(ord('A') + ind)
                landmark_id2name[label] = landmark
                plot_position = (landmark_img.shape[1]//2, 50)
                prompt_rgb_t = add_circle_to_image(
                    landmark_img.copy(),
                    [(plot_position, label)],
                    self.prompt_args,
                    info=info_cp,
                )
                prompt_rgb[:, ind*self.prompt_args['prompt_image_size'][1]:(ind+1)*self.prompt_args['prompt_image_size'][1], :] = prompt_rgb_t
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_{info["save_key"]}.png'))
            info_cp['landmark_id2name'] = landmark_id2name
            info_cp['summary'] = self.brief_summary

            num_retires = 3
            for i in range(num_retires):
                # get the landmark_id from the model
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

                landmark_id, return_info = self.get_param_from_response(response, info_cp)
                if landmark_id is not None:
                    break
            landmark = None
            if landmark_id is not None:
                landmark = landmark_id2name[landmark_id]
            else:
                landmark = None
        elif self.oracle_action:
            # ask user to select the landmark from the list of landmarks
            raise NotImplementedError

        capture_history = {
            'image': prompt_rgb,
            'query': query,
            'model_response': return_info['model_out'],
            'full_response': response,
            'landmark_id': landmark_id,
            'landmark': landmark,
            'error': return_info['error'],
            'model_analysis': '',
        }
        self.save_model_output(
            rgb=prompt_rgb,
            response=response,
            subtitles=[f'Task Query: {query}', f'Landmark ID: {landmark_id}'],
            img_file=os.path.join(self.vis_dir, f'output_{info["save_key"]}.png'),
        )

        error = None
        if len(return_info['error_list']) > 0:
            error = "Following errors have been produced: "
            for e in return_info['error_list']:
                error += f"{e}, "
            error = error[:-2]
            return self.on_failure(
                floor_num=floor_num,
                bld=bld,
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )

        landmark2poses = None
        if bld == 'ahg':
            landmark2poses = landmark2poses_floor1 if floor_num == 1 else landmark2poses_floor2
        elif bld == 'mbb':
            if floor_num == 1:
                landmark2poses = mbb_landmark2poses_floor1
            elif floor_num == 2:
                landmark2poses = mbb_landmark2poses_floor2
            elif floor_num == 3:
                landmark2poses = mbb_landmark2poses_floor3
            else:
                raise ValueError("Invalid floor number")
        elif bld == 'nhb':
            if floor_num == 3:
                landmark2poses = nhb_landmark2poses_floor3
            else:
                raise ValueError("Invalid floor number")
        # elif bld == 'nhb':
        #     if floor_num == 3:
        goal_pos_map, goal_ori_map = copy.deepcopy(landmark2poses[landmark])
        print(f"Moving to the landmark: {landmark}")
        print(f"Goal position: {goal_pos_map}, Goal orientation: {goal_ori_map}")

        is_success = False
        error = None
        if execute:
            user_input = input("Press Enter to continue or 'q' to quit: ")
            if user_input == 'q':
                execute = False
        if execute:
            print(colored("Move to the HOME position of arms?", 'red'))
            U.reset_env(env, reset_pose=RP.HOME_L_HOME_R, reset_pose_name='HOME_L_HOME_R', delay_scale_factor=1.0)
            goal = self.create_move_base_goal((goal_pos_map, goal_ori_map))
            state = self.send_move_base_goal(goal)
            if state == GoalStatus.SUCCEEDED:
                is_success = True
            else:
                error = movebase_code2error(state)
                is_success = False

        if not is_success:
            return self.on_failure(
                floor_num=floor_num,
                bld=bld,
                reason_for_failure=error,
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )
        if landmark == 'elevator':
            return self.on_failure(
                floor_num=floor_num,
                bld=bld,
                reason_for_failure=f'{return_info["error"]}. I navigated to the elevator instead to use elevator.',
                reset_required=False,
                capture_history=capture_history,
                return_info=return_info,
            )
        return self.on_success(
            floor_num=floor_num,
            bld=bld,
            capture_history=capture_history,
            return_info=return_info,
        )
