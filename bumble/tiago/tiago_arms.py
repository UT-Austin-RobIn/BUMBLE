import os
import numpy as np

import rospy
from std_msgs.msg import Header
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import WrenchStamped
from actionlib_msgs.msg import GoalID

from bumble.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener
from bumble.tiago.utils.transformations import euler_to_quat, quat_to_euler, add_angles, quat_to_rmat
from tracikpy import TracIKSolver
from scipy.spatial.transform import Rotation as R

class TiagoArms:

    def __init__(
            self,
            arm_enabled,
            side='right',
            torso_enabled=False,
            torso=None,
        ) -> None:
        self.arm_enabled = arm_enabled
        self.side = side

        self.torso = torso
        self.torso_enabled = torso_enabled
        self.ik_base_link = 'base_footprint' if torso_enabled else 'torso_lift_link'

        self.urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'urdf/tiago.urdf')

        self.setup_listeners()
        self.setup_actors()

    def setup_listeners(self):
        def joint_process_func(data):
            return np.array(data.actual.positions)

        self.arm_reader = TFTransformListener('/base_footprint')
        self.joint_reader = Listener(f'/arm_{self.side}_controller/state', JointTrajectoryControllerState, post_process_func=joint_process_func)
        def process_force(message):
            return message.wrench.force
        ### this is always for the right arm
        self.ft_right_sub = Listener(input_topic_name=f'/wrist_right_ft/corrected', input_message_type=WrenchStamped, post_process_func=process_force)

    @property
    def arm_pose(self):
        pos, quat = self.arm_reader.get_transform(target_link=f'/arm_{self.side}_tool_link')

        if pos is None:
            return None
        return np.concatenate((pos, quat))

    def setup_actors(self):
        self.arm_writer = None
        if self.arm_enabled:
            self.ik_solver = \
                TracIKSolver(
                    urdf_file=self.urdf_path,
                    base_link=self.ik_base_link,
                    tip_link=f"arm_{self.side}_tool_link",
                    timeout=4,
                    epsilon=1e-3,
                    solve_type="Distance"
                )
            self.arm_writer = Publisher(
                f'/arm_{self.side}_controller/safe_command',
                JointTrajectory
            )
            self.arm_cancel = Publisher(
                f'/arm_{self.side}_controller/follow_joint_trajectory/cancel',
                GoalID,
            )

    def process_action(self, action):
        # convert deltas to absolute positions
        pos_delta, euler_delta = action[:3], action[3:6]

        cur_pos, cur_quat = self.arm_reader.get_transform(
            target_link=f'/arm_{self.side}_tool_link',
            base_link=f'/{self.ik_base_link}'
        )
        cur_euler = quat_to_euler(cur_quat)
        target_euler = add_angles(euler_delta, cur_euler)
        target_pos = cur_pos + pos_delta
        print(f"Target Pos (wrt {self.ik_base_link}) {target_pos}; Target Euler (wrt {self.ik_base_link}) {target_euler}")
        target_quat = euler_to_quat(target_euler)
        return target_pos, target_quat

    def create_joint_command(self, joint_goal, duration_scale):
        message = JointTrajectory()
        message.header = Header()

        joint_names = []

        positions = list(self.joint_reader.get_most_recent_msg())
        for i in range(1, 8):
            joint_names.append(f'arm_{self.side}_{i}_joint')
            positions[i-1] = joint_goal[i-1]

        message.joint_names = joint_names

        # duration = 1.3
        duration = duration_scale # + 1.3
        point = JointTrajectoryPoint(positions=positions, time_from_start=rospy.Duration(duration))
        message.points.append(point)
        return message

    def is_at_joint(self, joint_goal, threshold=5e-3):
        cur_joints = self.joint_reader.get_most_recent_msg()
        return np.linalg.norm(cur_joints - joint_goal) < threshold

    def write(self, joint_goal, duration_scale, threshold=5e-3, delay_scale_factor=1.0, force_z_th=None):
        counter = 0
        if self.arm_writer is not None:
            while not self.is_at_joint(joint_goal, threshold):
                pose_command = self.create_joint_command(joint_goal, duration_scale)
                self.arm_writer.write(pose_command)
                force_vals = self.ft_right_sub.get_most_recent_msg()
                if force_z_th is not None:
                    assert self.side == 'right', "We only have force sensor for right arm."
                    print(f"force value: {force_vals.z} > {force_z_th}")
                    if force_vals.z < force_z_th: # we only use this for elevator.
                        print(f"Force value violated: {force_vals.z} < {force_z_th}")
                        # cancel pose command
                        joint_val = self.joint_reader.get_most_recent_msg()
                        pose_command = self.create_joint_command(joint_val, duration_scale=0.1)
                        self.arm_writer.write(pose_command)
                        rospy.sleep(1)
                        break
                counter += 1
                rospy.sleep(0.1)
                duration_scale = np.linalg.norm(joint_goal - self.joint_reader.get_most_recent_msg())*delay_scale_factor
        return counter

    def find_ik(self, target_pos, target_quat):
        ee_pose = np.eye(4)
        ee_pose[:3, :3] = quat_to_rmat(target_quat)
        ee_pose[:3, 3] = np.array(target_pos)

        joint_init = self.joint_reader.get_most_recent_msg()
        if self.torso_enabled:
            joint_init = np.concatenate((np.asarray([self.torso.get_torso_extension()]), joint_init))
        joint_goal = self.ik_solver.ik(ee_pose, qinit=joint_init)

        duration_scale = 0
        if joint_goal is not None:
            duration_scale = np.linalg.norm(joint_goal-joint_init)

        return joint_goal, duration_scale

    def step(self, action, delay_scale_factor=1.0, force_z_th=None):
        if self.arm_enabled:
            target_pos, target_quat = self.process_action(action)
            joint_goal, duration_scale = self.find_ik(target_pos, target_quat)
            duration_scale *= delay_scale_factor
            print("found joint_goal", joint_goal, "duraction_scale", duration_scale)

            if joint_goal is not None:
                if self.torso_enabled:
                    self.torso.torso_writer.write(self.torso.create_torso_command(joint_goal[0]))
                    joint_goal = joint_goal[1:]
                self.write(joint_goal, duration_scale, delay_scale_factor=delay_scale_factor, force_z_th=force_z_th)

            return {
                'joint_goal': joint_goal,
                'duration_scale': duration_scale
            }
        return {}

    def reset(self, action, allowed_delay_scale=4.0, delay_scale_factor=1.5, force_z_th=None):
        if self.arm_enabled:
            assert len(action) == 7

            cur_joints = self.joint_reader.get_most_recent_msg()
            delay_scale = np.linalg.norm(cur_joints - action)
            assert delay_scale < allowed_delay_scale, f"Resetting to a pose that is too far away: {delay_scale:.2f} > {allowed_delay_scale:.2f}"
            self.write(action, delay_scale*delay_scale_factor, delay_scale_factor=delay_scale_factor, force_z_th=force_z_th)
