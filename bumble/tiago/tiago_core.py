import time
import rospy

from bumble.tiago.grippers import PALGripper, RobotiqGripper2F_140, RobotiqGripper2F_85
from bumble.tiago.head import TiagoHead
from bumble.tiago.tiago_mobile_base import TiagoBaseVelocityControl
from bumble.tiago.tiago_torso import TiagoTorso
from bumble.tiago.tiago_arms import TiagoArms

class Tiago:
    gripper_map = {'pal': PALGripper, 'robotiq2F-140': RobotiqGripper2F_140, 'robotiq2F-85': RobotiqGripper2F_85}

    def __init__(self,
                    head_policy=None,
                    base_enabled=False,
                    torso_enabled=False,
                    right_arm_enabled=True,
                    left_arm_enabled=True,
                    right_gripper_type=None,
                    left_gripper_type=None):


        self.head_enabled = head_policy is not None
        self.base_enabled = base_enabled
        self.torso_enabled = torso_enabled

        self.head = TiagoHead(head_policy=head_policy)
        self.base = TiagoBaseVelocityControl(base_enabled=base_enabled)
        self.torso = TiagoTorso(torso_enabled=torso_enabled)
        self.arms = {
            'right': TiagoArms(right_arm_enabled, side='right', torso=self.torso, torso_enabled=False),
            'left': TiagoArms(left_arm_enabled, side='left', torso=self.torso, torso_enabled=False),
        }

        # set up grippers
        self.gripper = {'right': None, 'left': None}
        for side in ['right', 'left']:
            gripper_type = right_gripper_type if side=='right' else left_gripper_type
            if gripper_type is not None:
                self.gripper[side] = self.gripper_map[gripper_type](side)

        # default reset pose that it will go to after each episode
        self.reset_pose = {
                'right': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                'left': [0.43, -0.81, 1.60, 1.78, 1.34, -0.49, 1.15, 1],
                'torso': 0.15
            }
        # home pose
        # self.reset_pose = {
        #         'right': [-1.107, 1.466, 2.703, 1.718, -1.414, 1.391, 0.002, 1],
        #         'left': [-1.106, 1.466, 2.701, 1.729, -1.482, 1.385, 0.001, 1],
        #         'torso': 0.20
        #     }

    @property
    def right_gripper_pos(self):
        if self.gripper['right'] is None:
            return None
        return self.gripper['right'].get_state()

    @property
    def left_gripper_pos(self):
        if self.gripper['left'] is None:
            return None
        return self.gripper['left'].get_state()

    def step(self, action, delay_scale_factor=1.0, force_z_th=None):

        info = {}
        for side in ['right', 'left']:
            if action[side] is None:
                continue

            arm_action = action[side][:6]
            gripper_action = action[side][6]

            info[f'arm_{side}'] = self.arms[side].step(arm_action, delay_scale_factor=delay_scale_factor, force_z_th=force_z_th)

            if self.gripper[side] is not None:
                info[f'gripper_{side}'] = self.gripper[side].step(gripper_action)

        if self.head_enabled:
            info['head'] = self.head.step(action)

        if self.base_enabled:
            info['base'] = self.base.step(action['base'])

        if self.torso_enabled and (self.torso is not None) and ('torso' in action.keys()) and (action['torso'] is not None):
            self.torso.step(action['torso'])
        return info

    def reset(self, reset_arms=True, reset_pose=None, allowed_delay_scale=4.0, delay_scale_factor=1.5, wait_user=True, force_z_th=None):
        if reset_pose is None:
            reset_pose = self.reset_pose

        if ('torso' in reset_pose.keys()) and (self.torso is not None):
            self.torso.reset(reset_pose['torso'])

        for side in ['right', 'left']:
            if (reset_pose[side] is not None) and (self.arms[side].arm_enabled):
                if self.gripper[side] is not None:
                    self.gripper[side].step(reset_pose[side][-1])

                if reset_arms:
                    print(f'resetting {side}...{time.time()}')
                    self.arms[side].reset(reset_pose[side][:-1], allowed_delay_scale=allowed_delay_scale, delay_scale_factor=delay_scale_factor, force_z_th=force_z_th)
                    rospy.sleep(1)

        if self.head_enabled:
            self.head.reset_step(reset_pose)

        rospy.sleep(0.5)
        if wait_user:
            input('Reset complete. Press ENTER to continue')
        return True
