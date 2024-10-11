import os
import cv2
import copy
import pickle
import matplotlib.pyplot as plt
import numpy as np

import bumble
from bumble.tiago.skills.base import SkillBase
import bumble.utils.utils as U
from bumble.tiago.prompters.object_bbox import bbox_prompt_img

def make_prompt(skill_descs, task_desc, info=None, llm_baseline_info=None, method="ours"):
    """
    method arg is not being used for ablation results, just here for consistency
    """
    floor_num = info['floor_num']
    add_obj_ind = False
    bbox_ind2dist = None
    if info['add_obj_ind'] == True:
        add_obj_ind = True
        obj_bbox_list = info['obj_bbox_list']
        bbox_id2dist = info['bbox_id2dist']
        bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]
    skill_desc_str = ""
    for ind, skill_desc in enumerate(skill_descs):
        skill_desc_str += skill_desc + "\n\n"

    if llm_baseline_info:
        visual_instructions = [
            "Describe the distance to each of the objects.",
            "We describe the objects on the scene by their object id.",
            "Scene",
            "along with a description of the scene and visible objects",
            "analyzing the object descriptions",
        ]
    else:
        visual_instructions = [
            "First, describe the scene in the image. Describe each marked object briefly along with the distance to each of the object.",
            "The images are marked with object id.",
            "Scene in the image",
            "along with image of the scene",
            "looking at the objects in the image",
        ]

    instructions = f"""
INSTRUCTIONS:
You are given a description of the task that the robot must execute {visual_instructions[3]}. Given the current scene, break down the task into multiple sub-tasks to successfully complete the main task. {visual_instructions[0]} Describe where the robot is present in the building. Summarize the task that the robot must complete. If there was a failure in previous execution, add an explanation of why it might have failed by {visual_instructions[4]}. Next, create a plan of execution for the robot to follow to complete the task. Now, provide a short analysis of the sub-task that the robot must execute next and select the sub-task. In the analysis, include the verification step to check if the object can be manipulated (like pushing, picking) using the distance information provided and previous skill execution with additional verfication step:
Example 1: 'Object is 0.9 meters away from the robot, which is greater than 0.7 meters and hence cannot be grasped.'
Example 2: 'Object is 0.4 meters away but the pickup skill from previous time-step failed, hence the object cannot be grasped.'
Example 3: 'Object on the ground to be pushed is 1.5 meters away from the robot, which is less than 3 meters. Hence, the object is within the range of pushing.'
Be precise about the sub-task and describe the object(s) involved in the sub-task. Finally, using the sub-task, identitfy the skill that the robot must execute to complete the sub-task. You do NOT have to predict the arguments of the skill. Think carefully about the pre-conditions that must be satisfied before executing the skill. Verify if the object can be manipulated (like pushing, picking)
Select ONLY from the list of skills that the robot has and is feasible. The sub-task and skill_name are two different things. Sub-task is a high-level description of the task that the robot must complete. skill_name is the name of the skill that the robot must execute to complete the sub-task.
Provide the skill name in a valid JSON of this format
DESCRIPTION OF THE BUILDING:
    You are in the floor {floor_num} of the building. There are three floor. You can use the elevator to move between floors. The goto_landmark skill will select the elevator if the room is not found in the scene.
PLANNING EXAMPLES:
    Example 1: Get me something fruity to drink
        1. I am in the corridor of the hallway but drinks can be found inside the kitchen counter. Navigate to the kitchen where drinks are present.
        2. Since the soda cans are on the other side of the table, navigate closer to the fruity soda can present on the kitchen countertop using the navigate_to_point_on_ground skill.
        3. The fruity soda can is greater than 0.7 meters from the robot. I should move to the towards the fruity soda can on the table using the move skill.
        4. The fruity soda can still greater than 0.7 meters from the robot. I should move to the towards the fruity soda can on the table using the move skill for picking it up.
        5. Grasp the fruity soda can.
    Example 2: Get me a thriller book from the shelf
        1. Since I do not see any books on the table, I should go to the living room where bookshelf is usually present.
        2. Navigate to a point near the Agatha Christie book on the bookshelf.
        3. The thriller book, i.e., Agatha Christie book, user is interested in is still greater than 0.7 meters away from the robot. Use the move skill to move closer to the target book for picking it up.
        4. Grasp the book.
    Example 3: Get me a fruity drink from the fridge
        1. Go towards the kitchen where the refrigerator is present.
        2. There's one object blocking the path of the robot to reach refrigerator. Push the obstacle from the ground using the push_object_on_ground skill.
        3. Navigate to near the handle of the fridge using the navigate_to_point_on_ground skill.
        4. Open the fridge.
        5. Grasp the fruity drink from the comparment.
    Example 4: Can you bring me a duster?
        1. Classroom or seminar rooms have a whiteboard, where duster is usually present. Navigate to a classroom or seminar room with a whiteboard to get a duster.
        2. It looks like there are two obstacles blocking the way. Clear the path by pushing one obstacle from the ground using the push_object_on_ground skill.
        3. Now that atleast one object is pushed forward, the robot must continue to navigate to the classroom from the side.
        4. The table with the duster is further away from the robot. Navigate to near the table with the duster.
        5. Grasp the duster.
    Example 5: Get me something to eat that is creamy onion flavored
        1. I am in the corridor of the hallway but drinks can be found inside the kitchen or vending machine. Navigate to the kitchen where drinks can be found.
        2. Since the cream and onion chip packets are further away on the kitchen countertop, navigate near the cream and onion chips packet present on the table using the navigate_to_point_on_ground skill.
        3. The chips packet is greater than 0.7 meters from the robot. I should move to the towards the creamy onion chips packet on the table using the move skill for picking it up.
        4. The chips packet is still away from the robot. I should move to the towards the creamy onion chips packet can on the table using the move skill.
        5. Grasp the creamy onion chips packet.
    Example 6: Please arrange the drawer under my robot manipulator table
        1. I am in the corridor of the hallway but the task is to arrange the drawer in the robot manipulation room. Navigate to the manipulation room.
        2. Since the drawer is less than 3 meters away from the robot and close enough to push, push the drawer under the table using the push_object_on_ground skill without moving or navigating further.
    Example 7: Push the carboard box towards the wall
        1. Since the cardboard box is less than 3 meters away from the robot, the box is close enough to push. Simply push the cardboard box towards the wall using the push_object_on_ground skill to complete the task.
    Example 8: Can you bring me an HDMI to C-type cable to connect my laptop to the monitor?
        1. HDMI to C type cable can be found where there is projector setup, like a conference or class room. Navigate to the classroom or conference room where the projector setup is present to get a HDMI to C type cable.
        2. The table with the monitor setup is further away from the robot. Navigate to near the table with the HDMI to C type cable.
        3. Grasp the HDMI to C type cable.

Provide your answer at the end in a valid JSON of this format: {{"subtask": "", "skill_name": ""}}

The list of skills that the robot has are:
{skill_desc_str}
""".strip()
    instructions += f"""\n
GUIDELINES:
    - If the task involves in a particular area, always check if the robot is in the correct area. If not, navigate to the correct area.
    - If you do NOT see objects relevant to the task, describe potential locations where it can be found. Go to a location where the object can be found.
    - Avoid assuming objects if not present in the scene.
    - Avoid picking up objects that are too far away from the robot. Robot can only pick up objects that are approximately 0.7 meters. Always specify the distance of the object if selecting pick_up_object skill in the summary.
    - Avoid pushing objects that are too far away from the robot. Robot can only push objects that are withing 3.0 meters. Always specify the distance of the object if selecting push_object_on_ground skill in the summary. Note that the approximate distance to  push an object is within 3.0 meters but the distance to pick up an object is 0.7 meters.
    - Use navigate_to_point_on_ground skill to move to a point located on the ground near the object of interest to move inside the room, especially when the object is far away from the robot and ground is visible.
    - If it is already close to the object, use move skill instead of navigate_to_point_on_ground skill. Use move skill to adjust the robot's base position by few centimeters to reach the object.
    - Always pay attention to the information provided due to failure of skill execution.
    - If there is a failure due to collision, there may be an obstacle in front of the robot that may or may not be visible. Use push_object_on_ground skill to clear the path.
    - The push_object_on_ground will select the object and direction to push. All the objects may not be visible to you. Hence, providing names of the objects in the subtask can lead to selecting incorrect object by the push_object_on_ground skill.
    - If the collision is due to a door that you see, open the door using open_door skill before navigating further through the door.
    - Make sure to specify the destination floor in the subtask for call_elevator and use_elevator skills."""
    if add_obj_ind:
        instructions += f"""
    - {visual_instructions[1]} Below is the distance of each object with the robot. Use this information to decide the feasibility of manipulating the object.
    - Avoid using the object id in the final JSON response. Describe the object(s) involved in the sub-task instead of using the object id in the JSON response. This is very important.
OBSERVATIONS:"""
        for obj_id, dist in bbox_ind2dist:
            instructions += f"""
- Object id {obj_id} is {dist:.2f} meters from the robot."""
    instructions += f"""\n"""

    # Add scene description and obj descriptions if language only.
    if llm_baseline_info:
        instructions += f"""
- {visual_instructions[2]}: {llm_baseline_info['im_scene_desc']}
        """
        instructions += f"""
- Object ID descriptions: {llm_baseline_info['obj_descs']}"""

    task_prompt = f"""\nTASK DESCRIPTION: {task_desc}"""
    task_prompt += f"""\n

TIME-STEP: {info['step_idx']+1}
ANSWER: Let's think step by step.""".strip()
    return instructions, task_prompt

def make_history_prompt(history):
    instructions = f"""
Below is the execution history from previous time-steps of the same episode. Pay close attention to your previous predictions, success/failure feedback from the environment. Provide a summary of each of the errors in previous time-steps in your response. Avoid repeating the same errors. Based on the history, you can improve your predictions.
PREVIOUS TIME-STEP HISTORY:
""".strip()
    history_desc = []
    history_model_analysis = []
    for ind, msg in enumerate(history):
        example_desc = f"""\n
    TIME-STEP: {ind+1}
    DESCRIPTION: {msg['query']}
    ANSWER: {{"subtask": "{msg['model_response'][0]}", "skill_name": "{msg['model_response'][1]}"}}
    SKILL SUCCESS: {msg['is_success']}
    """.strip()
        if not msg['is_success']:
            example_desc += f"""\n
    FEEDBACK: {msg['env_reasoning']}
    """.strip()
        if ('model_analysis' not in msg) or (msg['model_analysis'] == ""):
            if not msg['is_success']:
                msg_to_add = "I made an error in my reasoning."
            else:
                msg_to_add = "The prediction is appropriate to complete the skill."
            msg['model_analysis'] = msg_to_add
        history_model_analysis.append(msg['model_analysis'])
        history_desc.append(example_desc)

    return instructions, history_desc, history_model_analysis

def make_cross_history_prompt(history):
    instructions = f"""
Below is the execution history from previous trials and not the current trial. The task may or may not be different from the current task. Pay close attention to the summary of the feedback. Avoid repeating the same errors. Based on the history, you can improve your predictions.
SUMMARY OF PREVIOUS TRIALS:
""".strip()
    history_desc = []
    history_model_analysis = []
    for ind, msg in enumerate(history):
        example_desc = f""
        if not msg['is_success']:
            example_desc += f"""\n
    SUMMARY {ind+1}: {msg['env_reasoning']}
    """.strip()

        if ('model_analysis' not in msg) or (msg['model_analysis'] == ""):
            if not msg['is_success']:
                msg_to_add = "I made an error in my reasoning."
            else:
                msg_to_add = "The prediction is appropriate to complete the skill."
            msg['model_analysis'] = msg_to_add
        history_model_analysis.append(msg['model_analysis'])
        history_desc.append(example_desc)

    return instructions, history_desc, history_model_analysis

class SkillSelector(SkillBase):
    def __init__(
        self,
        skill_descs: list[str],
        skill_names: list[str],
        run_dir: str,
        prompt_args: dict,
        add_histories: bool = False,
        reasoner_type: str = 'model',
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.skill_descs = skill_descs
        print(self.skill_descs)
        self.skill_names = skill_names
        self.n_vlm_evals = prompt_args.pop('n_vlm_evals', 0)
        self.add_obj_ind = prompt_args.pop('add_obj_ind', True)
        radius_per_pixel = prompt_args.pop('radius_per_pixel', 0.03)
        self.skill_name = 'selector'
        self.prompt_args = {
            "color": (0, 0, 0),
            "mix_alpha": 0.6,
            'thickness': 2,
            'rgb_scale': 255,
            'add_dist_info': prompt_args.get('add_dist_info', True),
            'add_object_boundary': prompt_args.get('add_object_boundary', False),
            'radius_per_pixel': radius_per_pixel,
        }

        self.vis_dir = os.path.join(run_dir, 'selector')
        os.makedirs(self.vis_dir, exist_ok=True)
        self.add_histories = add_histories
        self.reasoner_type = reasoner_type
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
        history_eval_dirs = None
        base_dir = os.path.join(bumble.__path__[0], 'long_term_mem', 'selector')
        history_eval_dirs = [os.path.join(base_dir, 'eval_id000.pkl')]
        return history_eval_dirs

    def create_language_history_msgs(
            self,
            history,
            func, # function to create the prompt
            func_kwargs, # kwargs for the function
            image_key=None,
        ):
        history_msgs = []
        history_inst, history_desc, history_model_analysis = func(history, **func_kwargs)
        history_imgs = [None] * len(history_desc)

        history_msgs = self.vlm.create_msg_history(
            history_instruction=history_inst,
            history_desc=history_desc,
            history_model_analysis=history_model_analysis,
            history_imgs=history_imgs,
        )
        return history_msgs

    def get_param_from_response(self, response, query, info):
        error_list = []
        return_info = {}
        return_info['response'] = response
        subtask = ''
        try:
            subtask = U.extract_json(response, 'subtask')
        except Exception as e:
            print(f"Error: {e}")
            subtask = query
            error = 'Missing subtask information in the JSON response.'
            error_list.append(error)

        skill_name = ''
        try:
            skill_name = U.extract_json(response, 'skill_name')
        except Exception as e:
            print(f"Error: {e}")
            skill_name = None
            error = 'Missing skill name in the JSON response.'
            error_list.append(error)

        if (skill_name is not None) and (skill_name not in self.skill_names):
            error = f"Skill name {skill_name} is not in the list of skills."
            error_list.append(error)

        return_info['error_list'] = error_list
        return_info['subtask'] = subtask
        return_info['skill_name'] = skill_name
        return subtask, skill_name, return_info

    def step(
        self,
        env,
        rgb,
        depth,
        pcd,
        normals,
        query,
        run_vlm=True,
        info=None,
        history=None,
        n_retries=3, # we query the model multiple times to avoid errors
        **kwargs,
    ):
        info = copy.deepcopy(info)
        step_idx = info['step_idx']
        e_value = 'incorrect'
        im = rgb.copy()
        img_size = min(im.shape[0], im.shape[1])

        self.prompt_args.update({
            'radius': int(img_size * self.prompt_args['radius_per_pixel']),
            'fontsize': int(img_size * 30 * self.prompt_args['radius_per_pixel']),
        })
        info.update({'add_obj_ind': self.add_obj_ind})

        if self.add_obj_ind:
            gsam_query = ['all objects']
            for _ in range(2):
                bboxes, mask_image = self.get_object_bboxes(rgb, query=gsam_query)
                if len(bboxes) > 0:
                    break
                else:
                    gsam_query = ['all objects and floor']
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
            info['obj_bbox_list'] = obj_bbox_list
            U.save_image(prompt_rgb, os.path.join(self.vis_dir, f'prompt_img_{info["save_key"]}.png'))
        else:
            prompt_rgb = rgb.copy()

        encoded_image = U.encode_image(prompt_rgb)
        history_msgs = None # this is for episodic history
        cross_history_msgs = None
        if self.add_histories:
            cross_history_msgs = self.create_history_msgs(
                self.history_list,
                func=make_cross_history_prompt,
                func_kwargs={},
            )
            history_msgs = cross_history_msgs
        if (history is not None) and (len(history)>0):
            ep_history_msgs = None
            if self.method == 'llm_baseline':
                ep_history_msgs = self.create_language_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                )
            else:
                ep_history_msgs = self.create_history_msgs(
                    history,
                    func=make_history_prompt,
                    func_kwargs={},
                )
            if history_msgs is None:
                history_msgs = ep_history_msgs
            else:
                history_msgs.extend(ep_history_msgs)

        for _ in range(n_retries):
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=history_msgs,
                make_prompt_func=make_prompt,
                make_prompt_func_kwargs={
                    'task_desc': query,
                    'skill_descs': self.skill_descs,
                    'info': info,
                }
            )
            #### creating the distance information string for capturing history
            bbox_ind2dist = [(bbox.obj_id, bbox.dist2robot) for bbox in obj_bbox_list]
            distance_str = ""
            for obj_id, dist in bbox_ind2dist:
                distance_str += f"""
- Object id {obj_id} is {dist:.2f} metres from the robot."""
            ####

            subtask, skill_name, return_info = self.get_param_from_response(response, query=query, info=info)
            capture_history = {
                'image': prompt_rgb,
                'query': query,
                'model_response': [subtask, skill_name],
                'full_response': response,
                'subtask': subtask,
                'skill_name': skill_name,
                'skill_info': self.skill_descs,
                'distance_info': distance_str,
                'model_analysis': '', # this will be added by an external evaluator
            }
            self.save_model_output(
                rgb=prompt_rgb,
                response=response,
                subtitles=[f'Task Query: {query}', f'Subtask: {subtask}\nSkill: {skill_name}'],
                img_file=os.path.join(self.vis_dir, f'output_{info["save_key"]}.png'),
            )
            if len(return_info['error_list']) == 0:
                break
        return_info.update({ # this will be reused in the pickup skill to avoid gsam queries
            'bboxes': bboxes,
            'mask_image': mask_image,
        })

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

        return self.on_success(
            capture_history=capture_history,
            return_info=return_info,
        )
