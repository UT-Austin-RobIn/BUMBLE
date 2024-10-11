import os
import copy
import cv2

import bumble.utils.utils  as U
from bumble.tiago.skills.base import SkillBase

def make_prompt_for_reasoning(task_desc, info, skill_name, *args, **kwargs):
    if skill_name == 'push_obstacle_on_ground':
        skill_name = 'push_object_on_ground'
    marker_info = info['marker_info']
    robot_info = info['robot_info']
    decision_template = info['decision_template']
    model_response = info['model_response']
    if 'floor_num' in info:
        floor_num = info['floor_num']
        if 'floor_num' in robot_info:
            robot_info = robot_info.format(floor_num=floor_num)
    if skill_name == 'selector':
        skill_info = info['skill_info']
        distance_info = info['distance_info']
        robot_info = robot_info.format(skill_info=skill_info, distance_info=distance_info)
        decision = decision_template.format(*model_response)

    if (skill_name == 'push_object_on_ground'):
        model_response = model_response[1:]
    decision = decision_template.format(*model_response) # this should be a list
    instructions = f"""
You are given an image, task description, and the decision made by the robot. Along with this, you are provided with an explanation of markers in the image, and capabilities of the robot.
Executing the decision by the robot led to a failure in the task. You need to analyze the task, scene, and the decision made by the robot to provide an analysis of the failure. Think about the collisions in the scenes, pressing wrong buttons, moving in an incorrect direction, distance to manipulated objects, etc. Avoid using marker ID like 'A', 'B', 'C' in your final answer to refer to the buttons, objects, or any scene information. Instead, use the description of the object, position, scene, etc. to provide a detailed analysis of the failure.

First, describe briefly the scene in front of you, all the objects, background, where the objects are placed, etc. Then, describe the intended task and its effect on the scene. Finally, analyze the decision made by the robot and provide a short analysis of why the decision led to a failure. Include why the robot failed in the short analysis. Make sure to provide an alternative decision that could have led to a successful task completion.

Provide your answer at the end in a valid JSON of this format: {{"short_analysis": []}}. Think step by step and finally provide your analysis in the JSON format.

An example of such an answer in JSON format include:
Example 1:
{{"short_analysis": ["The task was to move towards the BBQ flavored chips packet. In the scene, the chips packet is lying on a table in front of the robot along with other packets and drinks. The robot is less likely to collide with these objects as they placed on a table. However, the edge of the table is not visible, indicating that the robot is too close to the table. Given that the chips packet is on the right side of the robot, moving right would have avoided collision in the front and led to a successful task completion."]}}
Example 2:
{{"short_analysis": ["The task was to move towards the object placed on the extreme left side of the image. However, the robot moved in the right direction. This led to a failure in the task as the object was no longer in the robot's view. The robot should have moved in the left direction to reach the object."]}}
Example 3:
{{"short_analysis": ["The task was to call an elevator to the current floor for going to a higher floor. The robot pressed the emergency button with red colored marking. The emergency button is not used for calling the elevator. Among the remaining options, the robot should have pressed the button that is higher among the two other valid buttons marked in the image."]}}
Example 4:
{{"short_analysis": ["The task was to call an elevator to go the 4th floor. However, the robot selected a button which is marked with (OD) which typically indicates the open door button. The robot should have selected the button marked with (4) to go to the 4th floor."]}}
Example 5:
{{"short_analysis": ["There's a cardboad box in the scene to be pushed. The robot pushed the cardboard box in the left direction. However, there's a wall on the left side of the box. There's vending machine on the right side of the box. The robot could have pushed the object in the forward direction to avoid collision with the wall and the vending machine."]}}
Example 6:
{{"short_analysis": ["The scene consist of drawer, computer desk. The robot pushed the drawers in the right direction. The task was to push the drawers near the computer in front of it. The left and right directions do not allow the robot to complete the task. However, the robot could have pushed the drawer in the forward direction to move closer to the computer desk."]}}
Example 7:
{{"short_analysis": ["The robot must push the carboard box towards the desk. First, error is that the subtask included object id like 'B'. Instead, it should only include object description like, "the box lying in the middle of the room".
Second error is that despite the box being approximately 2 meters away from the robot, which is enough for pushing the object, the robot decided to adjust the robot base using 'move' skill. The robot should directly push the box.]}}
""".strip()

    prompt=f"""
TASK DESCRIPTION: {task_desc}
ROBOT CAPABILITIES: {robot_info}
EXPLANATION OF MARKERS: {marker_info}
ROBOT DECISION: {decision}
ANSWER: Let's think step by step.
""".strip()
    return instructions, prompt

class Reasoner(SkillBase):
    def __init__(
        self,
        *args, **kwargs
    ):
        self.skill_name2exp = {
            "move":
                {
                    "robot_info": "The robot can move its base in the specified directions. The direction can be either forward, backward, left, right w.r.t. the camera view. The forward direction is moving in the direction of the image (towards the top of the image), and backward is moving in the opposite direction. The left direction is moving to the left of the image, and right is moving to the right of the image. The robot uses its left arm to grab objects and can easily grasp objects on the left side of the image. If the robot moves right, the object will move to the left side of the image. If the robot moves left, the objects will move to the right side of the image. If the robot moves forward, the objects in the front will be closer and move to bottom of the image. If the robot moves backward, the objects in the front will move farther away from the robot, towards the top of the image.",
                    "marker_info": "Each object in the image is marked with an object id, example, 'B'. Along with the marked image, the image is marked with three directions: forward ('F'), left ('L'), and right ('R'). The points indicate the position of the robot if moved in that direction. Use this as an reference to analyze the scene and the robot's decision.",
                    "decision_template": "The robot moved in the {} direction."
                },
            "call_elevator":
                {
                    "robot_info": "The robot can press any of the marked buttons in the image to call the elevator to the current floor in order to go to a different floor. The robot is currently in floor {floor_num}. The task is to press the correct button to call the elevator to complete the task.",
                    "marker_info": "Each button-like object in the image is marked with an button ID, exmaple, 'B'. All the marked button-like object may not be buttons, example, emergency button, alarm button, key hole etc.",
                    "decision_template": "The robot pressed the button with ID {}.",
                },
            "use_elevator":
                {
                    "robot_info": "The robot can press any of the marked buttons in the image to use the elevator to go to a different floor. The task is to press the correct button to use the elevator to complete the task.",
                    "marker_info": "Each button-like object in the image is marked with an button ID, example, 'B'. You need to understand the task of each button, example, open door, door close buttons, floor selection buttons, etc.",
                    "decision_template": "The robot pressed the button with ID {}."
                },
            "push_object_on_ground":
                {
                    "robot_info": "The robot can push any obstacle on the ground to clear the path for the robot. The robot decides an object to push and the direction to push the object. The direction can be any of the three directions: forward, left, or right. While deciding the object to push, care should be taken to not push delicate objects or objects that can cause accidents later like stop sign. The direction to push the object should be to avoid collision with the other objects in the scene. To push the object in forward direction, the front of the object should be clear. Look for any tables, walls, or other objects in the front of the object. To push the object in the left direction, the left side of the object should be clear. Look for any objects on the left side of the object. To push the object in the right direction, the right side of the object should be clear. Look for any objects on the right side of the object.",
                    "marker_info": "The object robot pushed is marked with the object ID 'A'. Along with the marked image, the image is marked with three directions: forward ('F'), left ('L'), and right ('R'). The arrows indicate an approximate motion of the object for pushing in that direction. Use this as an reference to analyze the scene and the robot's decision.",
                    "decision_template": "The robot pushed the object with ID A in the {} direction.",
                },
            "selector":
                {
                    "robot_info": "The robot needs to select the subtask to best complete the task. The robot then needs to select one the skill that can help best achieve the task. The list of skills are provided below. It is important that the robot does not use object ID marked in the image in the description of the subtask predicted. Instead, the robot should use the description of the object, position, scene, etc. to select the subtask. The robot should not predict the parameters of the skill and specify only the skill name. The distances to the objects in the scene should be considered while selecting the skill, example, for picking up objects, the object must be within the reach of the robot, i.e, 0.7 meters. For pushing objects, the object must be within the reach of the robot, i.e, 3.0 meters. Include the distance information in the analysis. The robot should only select skill from the list of available skills.\n\nLIST OF SKILLS:\n{skill_info}\n\nDISTANCE TO OBJECTS:\n{distance_info}",
                    "marker_info": "Each object in the scene is marked with object id, example, object ID 'A'. Make sure that the subtask predicted does not include object ID marked in the image.",
                    "decision_template": "The robot predicted the next subtask: '{}'. The robot selected the skill: '{}'.",
                }

        }
        self.skill_name = 'reasoner'
        kwargs['skip_ros'] = True
        super().__init__(*args, **kwargs)

    def get_param_from_response(self, response, query, info):
        analysis = ''
        error_list = []
        return_info = {}
        return_info['response'] = response
        try:
            analysis = U.extract_json(response, 'short_analysis')
        except Exception as e:
            print(f"Error: {e}")
            analysis = None
            error = 'Missing analysis information in the JSON response.'
            error_list.append(error)
        return analysis, return_info

    def step(
        self,
        skill_name,
        history_i,
        info,
        *args, **kwargs
    ):
        is_success = history_i['is_success']
        if type(history_i['model_response']) is not list:
            history_i['model_response'] = [history_i['model_response']]
        info_cp = copy.deepcopy(info)
        assert skill_name in self.skill_name2exp, f"Skill {skill_name} not supported for reasoning yet. Please update."
        info_cp.update(self.skill_name2exp[skill_name])
        info_cp['model_response'] = history_i['model_response']

        n_retries = 3
        query = history_i['query']
        prompt_rgb = history_i['image']
        encoded_image = U.encode_image(prompt_rgb)
        for _ in range(n_retries):
            response = self.vlm_runner(
                encoded_image=encoded_image,
                history_msgs=None,
                make_prompt_func=make_prompt_for_reasoning,
                make_prompt_func_kwargs={
                    'skill_name': skill_name,
                    'task_desc': query,
                    'info': info_cp,
                }
            )
            analysis, return_info = self.get_param_from_response(response, query=query, info=info)
            if analysis is not None:
                break
        if analysis is None:
            analysis = ''
        return analysis, return_info
