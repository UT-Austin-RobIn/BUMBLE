# Description: Base class for all skills
import os
import sys
import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
from bumble.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
import bumble.utils.utils as U
import bumble.utils.transform_utils as T # transform_utils
from bumble.tiago.utils.transformations import quat_diff
from bumble.tiago.prompters.wrappers import GroundedSamWrapper
import bumble.tiago.prompters.vlms as vlms # GPT4V
from bumble.tiago.ros_restrict import set_init_pose

import actionlib
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from control_msgs.msg  import JointTrajectoryControllerState
from termcolor import colored

def movebase_code2error(status_code):
    status_dict = {
        GoalStatus.PENDING: ('PENDING', 'The goal has yet to be processed by the action server.'),
        GoalStatus.ACTIVE: ('ACTIVE', 'The goal is currently being processed by the action server.'),
        GoalStatus.PREEMPTED: ('PREEMPTED', 'The goal received a cancel request after it started executing and has since completed its execution (the goal was preempted).'),
        GoalStatus.SUCCEEDED: ('SUCCEEDED', 'The goal was achieved successfully by the action server.'),
        GoalStatus.ABORTED: ('ABORTED', 'The goal was aborted during execution by the action server due to collision at the goal pose.'),
        GoalStatus.REJECTED: ('REJECTED', 'The goal was rejected by the action server without being processed, because the goal was unattainable or invalid.'),
        GoalStatus.PREEMPTING: ('PREEMPTING', 'The goal received a cancel request before it started executing, but the action server has not yet confirmed that the goal is preempted.'),
        GoalStatus.RECALLING: ('RECALLING', 'The goal received a cancel request before it started executing and was recalled by the action server.'),
        GoalStatus.RECALLED: ('RECALLED', 'The goal received a cancel request before it started executing and was successfully cancelled (recalled).'),
        GoalStatus.LOST: ('LOST', 'The goal was sent by the action client but lost by the action server.')
    }
    return status_dict.get(status_code, ('UNKNOWN', 'Unknown status code.'))

class SkillBase:
    def __init__(
            self,
            vlm=None,
            tf_map=None,
            tf_odom=None,
            tf_base=None,
            tf_arm_left=None,
            tf_arm_right=None,
            client=None, # move_base client,
            head_pub=None,
            head_sub=None,
            init_pose_pub=None,
            method="ours",
            skip_ros=False,
        ):
        # Init all skills
        self.skip_ros = skip_ros
        self._gripper_correction_vec = np.array([0.19, -0.03, 0.0])

        if vlm is None:
            print("Initializing GPT4V model")
            self.vlm = vlms.GPT4V(openai_api_key=os.environ['OPENAI_API_KEY'])
            print("GPT4V model initialized")
        else:
            self.vlm = vlm

        self.default_head_joint_position = [0.0, -0.6]
        self._gsam = None
        self.init_pose_pub = init_pose_pub
        self.tf_map = tf_map
        self.tf_odom = tf_odom
        self.tf_base = tf_base
        self.tf_arm_left = tf_arm_left
        self.tf_arm_right = tf_arm_right
        self.client = client
        self.head_pub = head_pub
        self.head_sub = head_sub
        assert method in ["ours", "llm_baseline", "ours_no_markers"]
        self.method = method

    def set_gsam(self, gsam):
        self._gsam = gsam

    @property
    def gripper_correction_vec(self):
        return self._gripper_correction_vec

    def step(self, action):
        raise NotImplementedError

    def get_group_name(self, name):
        if name == 'left':
            return "arm_left"
        elif name == 'right':
            return "arm_right"
        else:
            raise ValueError(f'Invalid arm name: {name}')

    def setup_listeners(self):
        if self.skip_ros:
            return True
        print("setting up listeners")
        def joint_process_func(data):
            return np.array(data.actual.positions)

        if self.init_pose_pub is None:
            self.init_pose_pub = Publisher('/initialpose', PoseWithCovarianceStamped)
        if self.tf_map is None:
            self.tf_map = TFTransformListener('/map')
        if self.tf_odom is None:
            self.tf_odom = TFTransformListener('/odom')
        if self.tf_base is None:
            self.tf_base = TFTransformListener('/base_footprint')
        if self.tf_arm_left is None:
            self.tf_arm_left = TFTransformListener('/arm_left_tool_link')
        if self.tf_arm_right is None:
            self.tf_arm_right = TFTransformListener('/arm_right_tool_link')
        if self.client is None:
            print("waiting for move_base server")
            self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
            self.client.wait_for_server()
            print("move_base server found")
        if self.head_pub is None:
            self.head_pub = Publisher('/head_controller/command', JointTrajectory)
        if self.head_sub is None:
            def process_head(message):
                return message.actual.positions
            self.head_sub = Listener('/head_controller/state', JointTrajectoryControllerState, post_process_func=process_head)
        print("done setting up listeners")
        return True

    def left_arm_pose(self, frame):
        pos, quat = None, None
        if frame == 'odom':
            pos, quat = self.tf_odom.get_transform(target_link=f'/arm_left_tool_link')
        if frame == 'base_footprint':
            pos, quat = self.tf_base.get_transform(target_link=f'/arm_left_tool_link')
        if pos is None:
            return None
        return np.concatenate((pos, quat))

    def right_arm_pose(self, frame):
        pos, quat = None, None
        if frame == 'odom':
            pos, quat = self.tf_odom.get_transform(target_link=f'/arm_right_tool_link')
        if frame == 'base_footprint':
            pos, quat = self.tf_base.get_transform(target_link=f'/arm_right_tool_link')
        if pos is None:
            return None
        return np.concatenate((pos, quat))

    def extract_pcd_from_env_id(self, pcd, mask_image, env_id, filter_nan=True):
        mask = mask_image == env_id
        pcd_masked = pcd[mask]
        if filter_nan:
            mask = np.all(np.isnan(pcd_masked), axis=1)
            pcd_masked = pcd_masked[~mask]
        return pcd_masked

    def get_object_bboxes(self, rgb, query, gsam=None):
        # query is a list of object names
        if (self._gsam is None) and (gsam is None):
            self._gsam = GroundedSamWrapper(sam_ckpt_path=os.environ['SAM_CKPT_PATH'])
        if gsam is None:
            gsam = self._gsam
        bboxes = []
        final_mask_image = gsam.segment(rgb, query)
        final_mask_image = np.array(final_mask_image)
        unique_labels = np.unique(final_mask_image)

        for label in unique_labels:
            mask = final_mask_image == label
            # get the bbox of the mask
            nonzero = np.nonzero(mask)
            xmin, ymin, xmax, ymax = np.min(nonzero[1]), np.min(nonzero[0]), np.max(nonzero[1]), np.max(nonzero[0])
            bboxes.append([label, xmin, ymin, xmax, ymax])
        # sort the bboxes by the  x-coordinate and then y-coordinate
        bboxes = sorted(bboxes, key=lambda x: (x[1], x[2]))
        return bboxes, final_mask_image

    def send_head_command(self, head_positions, th=0.05):
        print("sending head command: ", head_positions)
        # write  joint trajectory ros message with positions as head_positions
        # head_positions is a list of 2 floats
        cur_head_position = np.asarray(self.head_sub.get_most_recent_msg())
        msg = JointTrajectory()
        msg.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = head_positions
        while np.linalg.norm(cur_head_position-head_positions) > th:
            print("sending head command: ", head_positions)
            point.time_from_start = rospy.Duration(0.5)
            msg.points.append(point)
            self.head_pub.write(msg)
            rospy.sleep(0.1)
            cur_head_position = np.asarray(self.head_sub.get_most_recent_msg())
        return True

    def send_move_base_goal(self, goal):
        print("sending move base goal")
        self.client.send_goal(goal)
        wait = self.client.wait_for_result()
        result = self.client.get_result()
        state = self.client.get_state()
        print("State from move_base: ", state)
        rospy.sleep(2) # the robot takes some time to reach the goal
        self.send_head_command(self.default_head_joint_position)
        return state

    def convert_gripper_pos2arm_pos(self, pos, side, frame):
        arm_pos = None
        # convert the gripper position to the tool_link frame
        transform = None
        if frame == 'odom':
            transform = T.pose2mat(self.tf_odom.get_transform(f'/arm_{side}_tool_link'))
        if frame == 'base_footprint':
            transform = T.pose2mat(self.tf_base.get_transform(f'/arm_{side}_tool_link'))
        transform_inv = np.linalg.inv(transform)
        pos_wrt_tool_link = (transform_inv @ np.concatenate((pos, np.array([1.0]))))[:3]
        # add the -ve of correction vector
        pos_wrt_tool_link = pos_wrt_tool_link - self.gripper_correction_vec # this can be different for each arm
        # convert it back to the base_footprint frame
        arm_pos = transform @ np.concatenate((pos_wrt_tool_link, np.array([1.0])))
        arm_pos = arm_pos[:3]
        return arm_pos

    def left_gripper_pos(self, frame):
        # read the average of the two finger pads
        pos, quat = None, None
        pos_wrt_tool_link = self.gripper_correction_vec
        if frame == 'odom':
            transform = T.pose2mat(self.tf_odom.get_transform('/arm_left_tool_link'))
        if frame == 'base_footprint':
            transform = T.pose2mat(self.tf_base.get_transform('/arm_left_tool_link'))
        pos = transform @ np.concatenate((pos_wrt_tool_link, np.array([1.0])))
        pos = pos[:3]
        return pos

    def right_gripper_pos(self, frame):
        pos, quat = None, None
        pos_wrt_tool_link = self.gripper_correction_vec
        if frame == 'odom':
            transform = T.pose2mat(self.tf_odom.get_transform('/arm_right_tool_link'))
        if frame == 'base_footprint':
            transform = T.pose2mat(self.tf_base.get_transform('/arm_right_tool_link'))
        pos = transform @ np.concatenate((pos_wrt_tool_link, np.array([1.0])))
        pos = pos[:3]
        return pos

    def close_gripper(self, env, side):
        assert side in ['left', 'right', 'both']
        if side == 'left':
            env.tiago.gripper['left'].step(0.0)
        if side == 'right':
            env.tiago.gripper['right'].step(0.0)
        if side == 'both':
            env.tiago.gripper['left'].step(0.0)
            env.tiago.gripper['right'].step(0.0)
        rospy.sleep(2)
        return

    def open_gripper(self, env, side):
        if side == 'left':
            env.tiago.gripper['left'].step(1.0)
        if side == 'right':
            env.tiago.gripper['right'].step(1.0)
        if side == 'both':
            env.tiago.gripper['left'].step(1.0)
            env.tiago.gripper['right'].step(1.0)
        rospy.sleep(2)
        return

    def create_move_base_goal(self, pose):
        goal_pos = pose[0]
        goal_ori = pose[1]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = goal_pos[0]
        goal.target_pose.pose.position.y = goal_pos[1]
        goal.target_pose.pose.position.z = goal_pos[2]
        goal.target_pose.pose.orientation.x = goal_ori[0]
        goal.target_pose.pose.orientation.y = goal_ori[1]
        goal.target_pose.pose.orientation.z = goal_ori[2]
        goal.target_pose.pose.orientation.w = goal_ori[3]
        return goal

    def localize_robot(self, floor, bld):
        set_init_pose(floor=floor, bld=bld, publisher=self.init_pose_pub)
        U.confirm_user(True, 'please check if there is roughly a good estimate of the robot in the map. y to continue (y/n). If not, then there is a bug')
        # this function should be called after changing the map
        # rotate along + z by + 90, then by -90 and original orientation:
        goal1_ori_base = np.asarray([0.0, 0.0, 0.7071068, 0.7071068])
        goal2_ori_base = np.asarray([0.0, 0.0, -0.7071068, 0.7071068])
        goal3_ori_base = np.asarray([0.0, 0.0, 0.0, 1.0])
        pos_base = np.asarray([0.0,0.0,0.0])
        goal1_pose = T.pose2mat((pos_base, goal1_ori_base))
        goal2_pose = T.pose2mat((pos_base, goal2_ori_base))
        goal3_pose = T.pose2mat((pos_base, goal3_ori_base))
        transform = T.pose2mat(self.tf_map.get_transform('/base_footprint'))

        goal1_pose_map = T.mat2pose(transform @ goal1_pose)
        goal2_pose_map = T.mat2pose(transform @ goal2_pose)
        goal3_pose_map = T.mat2pose(transform @ goal3_pose)

        goal1 = self.create_move_base_goal(goal1_pose_map)
        goal2 = self.create_move_base_goal(goal2_pose_map)
        goal3 = self.create_move_base_goal(goal3_pose_map)

        self.send_move_base_goal(goal1)
        self.send_move_base_goal(goal2)
        self.send_move_base_goal(goal3)

        return True

    def goto_odom_pos(self, env, pos, threshold=0.02, speed=0.5):
        # pos is the desired position in the odom frame
        speed_bangbang = speed

        transform = T.pose2mat(self.tf_odom.get_transform('/base_footprint'))
        diff = np.linalg.pinv(transform) @ np.concatenate((pos, np.array([1.0])))
        diff = np.abs(diff[:3])

        # write a closed loop controller to move the robot to the desired position
        while np.any(diff > threshold):
            control_input = np.zeros(3)
            # get the pos in current base_footprint frame
            transform = T.pose2mat(self.tf_odom.get_transform('/base_footprint'))
            diff = np.linalg.pinv(transform) @ np.concatenate((pos, np.array([1.0])))
            diff = diff[:3]

            # zero out components that are less than 0.01
            diff[np.abs(diff) < threshold] = 0.0
            control_input[diff > 0.0] = speed_bangbang
            control_input[diff < 0.0] = -speed_bangbang
            env.step({'base': control_input,'torso': None,'left': None,'right': None,'head': None})
            rospy.sleep(0.01)

            transform = T.pose2mat(self.tf_odom.get_transform('/base_footprint'))
            diff = np.linalg.pinv(transform) @ np.concatenate((pos, np.array([1.0])))
            diff = np.abs(diff[:3])

        # stop the robot
        env.step({'base': np.zeros(3),'torso': None,'left': None,'right': None,'head': None})
        rospy.sleep(0.1)
        return True

    def arm_goto_pose(
            self,
            env,
            pose,
            arm,
            frame,
            duration_scale_factor=1.0,
            gripper_act=None,
            adj_gripper=True,
            n_steps=2,
            force_z_th=None,
        ):
        '''
            THIS FUNCTION DOES NOT USE COLLISION CHECKING
                pose = (pos, ori) w.r.t. base_footprint
                gripper_act = 0.0 (close) or 1.0 (open)
                adj_gripper = True accounts for the final pose of the tip of the gripper
                n_steps = number of steps to interpolate between the current and final pose
        '''
        pose = copy.deepcopy(pose)
        final_pos = pose[0]
        if adj_gripper:
            final_pos = self.convert_gripper_pos2arm_pos(final_pos, arm, frame=frame)

        final_ori = pose[1]
        cur_arm_pose = self.left_arm_pose(frame=frame) if arm == 'left' else self.right_arm_pose(frame=frame)
        inter_pos = np.linspace(cur_arm_pose[:3], final_pos, n_steps)

        if gripper_act is None:
            gripper_act = env.tiago.gripper[arm].get_state()
            gripper_act = 0.0 if gripper_act < 0.5 else 1.0
            gripper_act = np.asarray([gripper_act])

        for pos in inter_pos[1:]:
            cur_arm_pose = self.left_arm_pose(frame=frame) if arm == 'left' else self.right_arm_pose(frame=frame)
            delta_pos = pos - cur_arm_pose[:3]
            delta_ori = R.from_quat(quat_diff(final_ori, cur_arm_pose[3:7])).as_euler('xyz')
            # if delta_ori is too small, then set it to zero
            delta_ori = delta_ori if np.linalg.norm(delta_ori) > 1e-3 else np.zeros(3)
            delta_act = np.concatenate(
                (delta_pos, delta_ori, gripper_act)
            )
            print(f'delta_pos: {delta_pos}', f'delta_ori: {delta_ori}')
            action = {'right': None, 'left': None, 'base': None}
            action[arm] = delta_act
            obs, reward, done, info = env.step(action, delay_scale_factor=duration_scale_factor, force_z_th=force_z_th)
        return obs, reward, done, info

    def create_history_msgs(
            self,
            history,
            func, # function to create the prompt
            func_kwargs, # kwargs for the function
            image_key='image',
        ):
        history_msgs = []
        history_inst, history_desc, history_model_analysis = func(history, **func_kwargs)
        history_imgs = []
        for msg in history:
            assert image_key in msg
            # check if the image is already encoded
            assert isinstance(msg[image_key], np.ndarray), f"Image is not a numpy array, but {type(msg[image_key])}"
            # encode the image. First convert to bgr and then encode.
            encoded_image = U.encode_image(msg[image_key])
            history_imgs.append(encoded_image)

        history_msgs = self.vlm.create_msg_history(
            history_instruction=history_inst,
            history_desc=history_desc,
            history_model_analysis=history_model_analysis,
            history_imgs=history_imgs,
        )
        return history_msgs

    def on_failure(
            self,
            reason_for_failure: str,
            reset_required: bool,
            capture_history: dict, # history used for future prompting
            return_info: dict
        ):
        return_info['reset_required'] = reset_required
        capture_history['env_reasoning'] = reason_for_failure
        capture_history['is_success'] = False
        # print(f"Failed: {reason_for_failure}")
        print(colored(f"Failed: {reason_for_failure}", 'red'))
        print(colored(f"Reset Required: {reset_required}", 'red'))
        return False, reason_for_failure, capture_history, return_info

    def tuck_in_gripper(self, env, arm):
        duration_scale_factor = 0.5
        cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
        goal_joint_angles = copy.deepcopy(cur_joint_angles)
        self.tuck_out_joint_angles = copy.deepcopy(cur_joint_angles)
        goal_joint_angles[-2] = -0.28
        duration_scale = duration_scale_factor*np.linalg.norm(goal_joint_angles-cur_joint_angles) # move twice as low
        env.tiago.arms[arm].write(goal_joint_angles, duration_scale, delay_scale_factor=duration_scale_factor)
        return True

    def tuck_out_gripper(self, env, arm):
        duration_scale_factor = 0.5
        cur_joint_angles = env.tiago.arms[arm].joint_reader.get_most_recent_msg()
        goal_joint_angles = self.tuck_out_joint_angles
        duration_scale = duration_scale_factor*np.linalg.norm(goal_joint_angles-cur_joint_angles)
        env.tiago.arms[arm].write(goal_joint_angles, duration_scale, delay_scale_factor=duration_scale_factor)
        self.tuck_out_joint_angles = None
        return True

    def on_success(
            self,
            capture_history: dict,
            return_info: dict
        ):
        return_info['reset_required'] = False
        capture_history['env_reasoning'] = ''
        capture_history['is_success'] = True
        print(colored("Success", 'green'))
        return True, None, capture_history, return_info

    def scene_prompt_func(self):
        """Used for baseline to provide textual description of image"""
        instructions = """
INSTRUCTIONS:
You will be given an image of the scene. First, describe the scene in the image. Then, describe each marked object briefly.
Provide all the descriptions at the end in a valid JSON of this format: {{"scene_description": "", "obj_descriptions", ""}}"""
        task_prompt = """
ANSWER: Let's think step-by-step."""
        return instructions, task_prompt

    def get_param_from_scene_obj_resp(self, response):
        error_list = []
        return_info = {}
        return_info['response'] = response

        scene_desc = ''
        try:
            scene_desc = U.extract_json(response, 'scene_description')
        except Exception as e:
            print(f"Error: {e}")
            error = 'Missing scene description information in the JSON response.'
            error_list.append(error)

        obj_descs = ''
        try:
            obj_descs = U.extract_json(response, 'obj_descriptions')
        except Exception as e:
            print(f"Error: {e}")
            obj_descs = None
            error = 'Missing skill name in the JSON response.'
            error_list.append(error)
        if isinstance(obj_descs, dict):
            obj_id2desc_map = dict(obj_descs)
            obj_descs = ""
            for _id in sorted(obj_id2desc_map.keys()):
                obj_descs += f"{_id}: {obj_id2desc_map[_id]} "
            obj_descs = obj_descs.strip()

        return_info['error_list'] = error_list
        return_info['scene_desc'] = scene_desc
        return_info['obj_descs'] = obj_descs
        return scene_desc, obj_descs, return_info

    def prep_llm_prompt(
            self,
            encoded_image,
            make_prompt_func,
            make_prompt_func_kwargs):
        # First get a textual description of the scene and object IDs from image.
        instructions, task_prompt = self.scene_prompt_func()
        prompt_seq = [task_prompt, encoded_image]
        scene_obj_desc_response = self.vlm.query(instructions, prompt_seq)
        print(scene_obj_desc_response)
        scene_desc, obj_descs, scene_return_info = (
            self.get_param_from_scene_obj_resp(scene_obj_desc_response))

        # Update make_prompt_func_kwargs for actual selector/skill call
        llm_baseline_prompt_info = dict(
            im_scene_desc=scene_desc,
            obj_descs=obj_descs,
        )
        make_prompt_func_kwargs.update(dict(
            llm_baseline_info=llm_baseline_prompt_info,
            method=self.method,
        ))
        instructions, task_prompt = make_prompt_func(**make_prompt_func_kwargs)
        prompt_seq = [task_prompt]
        return instructions, prompt_seq

    def vlm_runner(
        self,
        encoded_image,
        history_msgs,
        make_prompt_func,
        make_prompt_func_kwargs,
        force_vlm_prompt=False,
    ):
        if self.method == "llm_baseline" and not force_vlm_prompt:
            instructions, prompt_seq = self.prep_llm_prompt(
                encoded_image,
                make_prompt_func,
                make_prompt_func_kwargs)
            task_prompt = "".join(prompt_seq)
        else:
            make_prompt_func_kwargs.update(dict(method=self.method))
            instructions, task_prompt = make_prompt_func(**make_prompt_func_kwargs)
            prompt_seq = [task_prompt, encoded_image]
        if self.method != "ours":
            print(colored(f"{self.method} Prompt\n" + instructions + task_prompt, 'light_blue'))
        response = self.vlm.query(instructions, prompt_seq, history=history_msgs)
        print(f"************************* {self.skill_name} ******************************")
        print(colored(response, 'yellow'))
        return response

    def save_model_output(
        self,
        rgb,
        response,
        subtitles,
        img_file,
    ):
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].imshow(rgb); ax[0].axis('off')
        ax[0].set_title(subtitles[0])
        import textwrap
        text = response
        textwrap = textwrap.TextWrapper(width=75)
        text = textwrap.fill(text)
        ax[1].imshow(np.ones_like(rgb) * 255); ax[1].axis('off')
        ax[1].text(0, 0, text, fontsize=10, color='black', wrap=True)
        ax[1].set_title(subtitles[1])
        ax[1].set_xlim(0, rgb.shape[1])
        ax[1].set_ylim(0, rgb.shape[0])
        plt.savefig(img_file)
        plt.clf()
        return
