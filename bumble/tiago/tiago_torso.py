import numpy as np

import rospy
from std_msgs.msg import Header
from control_msgs.msg  import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from bumble.tiago.utils.ros_utils import Publisher, Listener

class TiagoTorso:

    def __init__(self, torso_enabled) -> None:
        self.torso_enabled = torso_enabled

        self.setup_listener()
        self.setup_actors()

    def setup_listener(self):
        def process_torso_state(message):
            return message.actual.positions[0]

        self.torso_listener = Listener(input_topic_name='/torso_controller/state', input_message_type=JointTrajectoryControllerState, post_process_func=process_torso_state)

    def setup_actors(self):
        self.torso_writer = Publisher('/torso_controller/safe_command', JointTrajectory)

    def create_torso_command(self, dist):
        message = JointTrajectory()
        message.header = Header()
        message.joint_names = ['torso_lift_joint']
        point = JointTrajectoryPoint(positions=[dist], time_from_start=rospy.Duration(0.5))
        message.points.append(point)

        return message

    def is_at_joint(self, joint_goal, threshold=0.01):
        print("inside torso is at joint")
        current_joint = self.torso_listener.get_most_recent_msg()
        print(current_joint, joint_goal)
        return np.abs(current_joint - joint_goal) < threshold

    def step(self, position_delta):

        if self.torso_enabled and (position_delta is not None):
            goal_position = np.clip(self.get_torso_extension() + position_delta, 0.05, 0.30)
            self.torso_writer.write(self.create_torso_command(goal_position))

    def get_torso_extension(self):
        current_torso_extension = self.torso_listener.get_most_recent_msg()
        return current_torso_extension

    def reset(self, abs_pos, threshold=0.01):
        current_joint = self.torso_listener.get_most_recent_msg()
        if abs_pos is not None:
            abs_pos = min(0.35, abs_pos)
            while abs(current_joint-abs_pos) > threshold:
                self.torso_writer.write(self.create_torso_command(abs_pos))
                current_joint = self.torso_listener.get_most_recent_msg()
        return current_joint

