import os
import copy
import pickle
import argparse
import numpy as np
from easydict import EasyDict
from termcolor import colored

import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from control_msgs.msg  import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import bumble.utils.utils as U
import bumble.utils.transform_utils as T # transform_utils
import bumble.utils.vision_utils as VU # vision_utils

from bumble.tiago.tiago_gym import TiagoGym
from bumble.tiago.utils.camera_utils import Camera
from bumble.tiago.skills.selector import SkillSelector
import bumble.tiago.RESET_POSES as RP
from bumble.tiago.skills import MoveToSkill, PickupSkill, GoToLandmarkSkill, UseElevatorSkill, OpenDoorSkill, PushObsGrSkill, NavigateToPointSkill, CallElevatorSkill
import bumble.tiago.prompters.vlms as vlms # GPT4V
from bumble.tiago.prompters.wrappers import GroundedSamWrapper
from bumble.tiago.ros_restrict import change_map, set_floor_map
from bumble.tiago.utils.ros_utils import Publisher, Listener, TFTransformListener

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
rospy.init_node('tiago_test')

def load_skill(skill_id, args, kwargs_to_add):
    if skill_id == 'move':
        prompt_args = {
            'raidus_per_pixel': 0.06,
            'arrow_length_per_pixel': 0.1,
            'plot_dist_factor': 1.0,
            'plot_direction': True,
        }
        skill = MoveToSkill(
            oracle_action=args.oracle,
            debug=False,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            add_histories=args.add_past,
            **kwargs_to_add,
        )
    elif skill_id == 'pick_up_object':
        prompt_args = {
            'add_object_boundary': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.04,
        }
        skill = PickupSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'goto_landmark':
        prompt_args = {
            'raidus_per_pixel': 0.03,
        }
        skill = GoToLandmarkSkill(
            bld=args.bld,
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'open_door':
        prompt_args = {}
        skill = OpenDoorSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    elif skill_id == 'use_elevator':
        prompt_args = {
            'add_object_boundary': False,
            'add_dist_info': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.03,
        }
        skill = UseElevatorSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            add_histories=args.add_past,
            **kwargs_to_add,
        )
    elif skill_id == 'call_elevator':
        prompt_args = {
            'add_object_boundary': False,
            'add_dist_info': False,
            'add_arrows_for_path': False,
            'radius_per_pixel': 0.03,
        }
        skill = CallElevatorSkill(
            oracle_position=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            add_histories=args.add_past,
            **kwargs_to_add,
        )
    elif skill_id == 'push_obs_gr':
        prompt_args = {}
        skill = PushObsGrSkill(
            oracle_action=args.oracle,
            debug=True,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            add_histories=args.add_past,
            **kwargs_to_add,
        )
    elif skill_id == 'navigate_to_point_gr':
        prompt_args = {
            'raidus_per_pixel': 0.04,
            'arrow_length_per_pixel': 0.1, # don't need this
            'plot_dist_factor': 1.0, # don't need this
        }
        skill = NavigateToPointSkill(
            oracle_action=args.oracle,
            debug=args.debug,
            run_dir=args.run_dir,
            prompt_args=prompt_args,
            **kwargs_to_add,
        )
    else:
        raise ValueError(f"Unknown skill id: {skill_id}")
    return skill

def update_history(
        is_success,
        reason_for_failure,
        history_i,
        args,
    ):
    print(colored(f"Success: {is_success}", 'green' if is_success else 'red'))
    history_i['is_success'] = is_success
    if is_success:
        return history_i

    history_i['env_reasoning'] = reason_for_failure
    return history_i

def get_kwargs_to_add():
    print("Loading VLM")
    vlm = vlms.GPT4V(openai_api_key=os.environ['OPENAI_API_KEY'])
    print("Done.")
    print("Loading transforms")
    tf_map = TFTransformListener('/map')
    tf_odom = TFTransformListener('/odom')
    tf_base = TFTransformListener('/base_footprint')
    tf_arm_left = TFTransformListener('/arm_left_tool_link')
    tf_arm_right = TFTransformListener('/arm_right_tool_link')
    print("Done.")
    print("Loading action client for move_base")
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()
    print("Done")
    head_pub = Publisher('/head_controller/command', JointTrajectory)
    def process_head(message):
        return message.actual.positions
    head_sub = Listener('/head_controller/state', JointTrajectoryControllerState, post_process_func=process_head)
    kwargs_to_add = {
        'vlm': vlm,
        'tf_map': tf_map,
        'tf_odom': tf_odom,
        'tf_base': tf_base,
        'tf_arm_left': tf_arm_left,
        'tf_arm_right': tf_arm_right,
        'client': client,
        'head_pub': head_pub,
        'head_sub': head_sub,
    }
    return kwargs_to_add

def get_task_query_poses(eval_id, args):
    task_query, default_h_r, default_name = None, None, None
    query_list = []
    if eval_id == 1:
        query_list = ["I want to color the sky in my drawing. Can you get me a marker?", "I want to color grass in my drawing. Can you get me a marker?", "I need to color some hearts. Can you get a marker for that?"]
        default_h_r = RP.PICKUP_TABLE_L_HOME_R
        default_name = 'PICKUP_TABLE_L_HOME_R'
        default_h_r['torso'] = 0.15 # tables in the building are too low.
    elif eval_id == 2:
        query_list = ["Could you grab me a drink that is low in calories?", "Any chance you can find me a sugar-free soda?", "I want something fizzy to drink, but I am on diet. Can you help me with that?"]
        default_h_r = RP.PICKUP_TABLE_L_HOME_R_H
        default_name = 'PICKUP_TABLE_L_HOME_R_H'
    elif eval_id == 3:
        query_list = ["Could you make the seating chairs in the reception area more orderly?", "Make the reception area more welcoming by arranging the chairs", "Can you help me arrange the chairs in the reception area?"]
        default_h_r = RP.HOME_L_PUSH_R_H
        default_name = 'HOME_L_PUSH_R_H'
    else:
        raise NotImplementedError("This task is not implemented yet.")
    return query_list, default_h_r, default_name

def main(args):
    run_dir = args.run_dir
    kwargs_to_add = get_kwargs_to_add()
    kwargs_to_add['method'] = args.method
    if run_dir is None:
        skill_str = '_'.join(args.skills)
        run_dir = f'{args.bld}_eval_id{args.eval_id}_{args.floor_num}_vlm{args.run_vlm}_select_hist{args.add_selection_history}_past{args.add_past}'
        if args.suffix is not None:
            run_dir = f'{run_dir}_{args.suffix}'
        run_dir = os.path.join('experiments', run_dir)
        # check all directories starting with run_XXX in run_dir
        dir_ind = 0
        if os.path.exists(run_dir): # if the directory already exists check the last run number
            dir_ind = len([d for d in os.listdir(run_dir) if d.startswith('run_')])
        run_dir = os.path.join(run_dir, f'run_{dir_ind:03d}')
        args.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

    # print in the terminal the run_dir
    print(colored("-"*50, 'green'))
    print(colored(f"Run directory: {run_dir}", 'green'))
    print(colored("-"*50, 'green'))

    # move_base will not work without publishing this empty map.
    floor_num = args.floor_num
    set_floor_map(floor_num, bld=args.bld)
    pid = change_map(floor_num=floor_num, bld=args.bld, empty=False) # this one will only add he additional map.

    print("Loading gsam")
    gsam = None
    if args.run_vlm:
        # make sure the head is -0.8
        gsam = GroundedSamWrapper(sam_ckpt_path=os.environ['SAM_CKPT_PATH'])
    print("Gsam loading done")

    skill_id_list = args.skills
    skill_list = []
    for skill_id in skill_id_list:
        skill_list.append(load_skill(skill_id, args, kwargs_to_add=kwargs_to_add))
    skill_name2obj = {}
    skill_descs = []
    for skill in skill_list:
        skill.set_gsam(gsam)
        skill_name2obj[f'{skill.skill_name}'] = skill
        skill_descs.append(skill.skill_descs)

    print("Loading gym")
    env = TiagoGym(
        frequency=10,
        right_arm_enabled=True,
        left_arm_enabled=True,
        right_gripper_type='robotiq2F-140',
        left_gripper_type='robotiq2F-85',
        base_enabled=True,
        torso_enabled=True,
    )
    print("Gym loading done.")
    prompt_args = {
        'n_vlm_evals': 0,
        'add_obj_ind': True,
        'raidus_per_pixel': 0.04,
        'add_dist_info': True,
        'add_object_boundary': False,
    }

    log_dir = os.path.join(args.run_dir, 'logs')
    video_dir = os.path.join(args.run_dir, 'videos')
    ss_dir = os.path.join(args.run_dir, 'screenshots')
    data_dir = os.path.join(args.run_dir, 'data')
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(ss_dir, exist_ok=True)
    config = EasyDict(args.__dict__)
    U.save_yaml(config, os.path.join(run_dir, 'config.yaml'))

    selector_skill = SkillSelector(
        skill_descs=skill_descs,
        skill_names=skill_name2obj.keys(),
        run_dir=run_dir,
        prompt_args=prompt_args,
        add_histories=args.add_past,
        reasoner_type='model',
        **kwargs_to_add,
    )
    selector_skill.set_gsam(gsam)
    selector_skill.send_head_command(selector_skill.default_head_joint_position)
    logger_dict = {'success': [], 'skill_selection_list': [], 'eval_ind': [], 'reason_for_failure': [], 'describe_traj': [], 'completed_subtasks': [], 'total_subtasks': []}
    logger_dict = EasyDict(logger_dict)
    start_ind = 0

    #### Initializing saving variables
    for eval_ind in range(start_ind, args.n_eval, 1):
        U.clear_input_buffer()
        input("Press Enter to start the evaluation. Reset the robot to the initial pose.")

        query_list, grasp_h_r, name = get_task_query_poses(eval_id=args.eval_id, args=args)
        for i, query in enumerate(query_list):
            print(colored(f"Query {i+1}: {query}", 'green'))
        U.clear_input_buffer()
        select_query = None
        while not (select_query in [str(i+1) for i in range(len(query_list))]):
            select_query = input("Select the query number: ")
        task_query = query_list[int(select_query)-1]

        default_h_r = copy.deepcopy(grasp_h_r)
        default_name = name

        traj_save = {}
        traj_save['cam_intr'] = []
        traj_save['cam_extr'] = []
        skill_histories = {'selection': []}
        for skill_name in skill_name2obj.keys():
            skill_histories[skill_name] = []
        traj_success = False

        tf_listener = TFTransformListener('/base_footprint')

        #### Start the main loop
        U.reset_env(env, reset_pose=grasp_h_r, reset_pose_name=name, delay_scale_factor=1.5)

        start_select_ind = 0
        selection_seq = []

        for select_ind in range(start_select_ind, args.n_skill_selection, 1):
            # check if it is safe to move to the reset pose
            # print the name of the reset pose and value in red
            U.clear_input_buffer()
            U.reset_env(env, reset_pose=grasp_h_r, reset_pose_name=name, delay_scale_factor=1.5)

            #### Skill selection
            obs_pp = VU.get_obs(env, tf_listener)
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']

            ################### storing data for reproducibility
            U.save_image(rgb, os.path.join(ss_dir, f'rgb_eval_id{eval_ind:03d}_skill_ind{select_ind:03d}.png'))
            U.save_image(depth.astype(np.uint8), os.path.join(ss_dir, f'depth_eval_id{eval_ind:03d}_skill_ind{select_ind:03d}.png'))
            ##################

            save_key = f'eval_id_{eval_ind:03d}_step_{select_ind:03d}'
            info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': eval_ind, 'save_key': save_key, 'floor_num': floor_num}
            is_success, selection_error, selection_history, selection_return_info = \
                    selector_skill.step(
                        env=env,
                        rgb=rgb,
                        depth=depth,
                        pcd=pcd,
                        normals=normals,
                        query=task_query,
                        arm=args.arm,
                        execute=args.exec,
                        run_vlm=args.run_vlm,
                        info=info,
                        history=skill_histories['selection'] if args.add_selection_history else None,
                    )

            #### Update history
            history_i = update_history(
                is_success=is_success,
                reason_for_failure=selection_error,
                history_i=selection_history,
                args=args
            )
            skill_histories['selection'].append(history_i)
            if not is_success:
                continue

            skill_name = selection_return_info['skill_name']
            subtask = selection_return_info['subtask']
            selection_seq.append(skill_name)

            #### Skill execution
            obs_pp = VU.get_obs(env, tf_listener)
            rgb, depth, cam_intr, cam_extr, pcd, normals = obs_pp['rgb'], obs_pp['depth'], obs_pp['cam_intr'], obs_pp['cam_extr'], obs_pp['pcd'], obs_pp['normals']
            save_key = f'eval_id_{eval_ind:03d}_step_{select_ind:03d}'
            info = {'step_idx': select_ind, 'cam_intr': cam_intr, 'cam_extr': cam_extr, 'eval_ind': eval_ind, 'save_key': save_key, 'floor_num': floor_num}

            is_success, reason_for_failure, history, return_info = None, None, None, None
            common_args = {
                'env': env,
                'rgb': rgb,
                'depth': depth,
                'pcd': pcd,
                'normals': normals,
                'arm': args.arm,
                'info': info,
                'history': None,
                'query': subtask, # or task_query
            }
            if skill_name == 'move':
                is_success, reason_for_failure, history, return_info = \
                        skill_name2obj[skill_name].step(
                            **common_args,
                            execute=args.exec,
                            run_vlm=args.run_vlm,
                            debug=False,
                            mask_image=selection_return_info['mask_image'], # avoids re-querying the gsam
                            bboxes=selection_return_info['bboxes'], # avoids re-querying the gsam
                        )
            elif skill_name == 'pick_up_object':
                common_args['arm'] = 'left'
                is_success, reason_for_failure, history, return_info = \
                        skill_name2obj[skill_name].step(
                            **common_args,
                            execute=args.exec,
                            run_vlm=args.run_vlm,
                            debug=args.debug,
                            mask_image=selection_return_info['mask_image'], # avoids re-querying the gsam
                            bboxes=selection_return_info['bboxes'], # avoids re-querying the gsam
                        )
            elif skill_name == 'goto_landmark':
                is_success, reason_for_failure, history, return_info = \
                        skill_name2obj[skill_name].step(
                            **common_args,
                            execute=args.exec,
                            run_vlm=args.run_vlm,
                            debug=args.debug,
                            floor_num=floor_num,
                            bld=args.bld,
                        )
            elif skill_name == 'open_door':
                # this aligns the robot with the door
                skill_name2obj[skill_name].align_with_door(env, floor_num=floor_num, arm=args.arm, execute=args.exec)
                common_args['arm'] = 'right' # always done with the right arm. Safer.
                # the skill itself will put the correct arm in the correct position
                is_success, reason_for_failure, history, return_info = \
                        skill_name2obj[skill_name].step(
                            **common_args,
                            execute=args.exec,
                            run_vlm=args.run_vlm,
                            debug=args.debug,
                        )
            elif skill_name == 'push_object_on_ground':
                is_success, reason_for_failure, history, return_info = \
                        skill_name2obj[skill_name].step(
                            **common_args,
                            execute=args.exec,
                            run_vlm=args.run_vlm,
                            debug=args.debug,
                            mask_image=selection_return_info['mask_image'], # avoids re-querying the gsam
                            bboxes=selection_return_info['bboxes'], # avoids re-querying the gsam
                            floor_num=floor_num,
                            bld=args.bld,
                        )
            elif (skill_name == 'use_elevator') or (skill_name == 'call_elevator'):
                U.reset_env(env, reset_pose=RP.HOME_L_PUSH_R_H, reset_pose_name='HOME_L_PUSH_R_H', delay_scale_factor=1.5)
                grasp_h_r = RP.HOME_L_PUSH_R_H
                name = 'HOME_L_PUSH_R_H'
                common_args['arm'] = 'right' # always done with the right arm. Safer.
                is_success, reason_for_failure, history, return_info = \
                        skill_name2obj[skill_name].step(
                            **common_args,
                            execute=args.exec,
                            run_vlm=args.run_vlm,
                            debug=args.debug,
                            bld=args.bld,
                        )
                if skill_name == 'use_elevator':
                    floor_num = return_info['floor_num'] # change in the floor number
                    set_floor_map(floor_num, bld=args.bld)
                    pid = change_map(floor_num=floor_num, bld=args.bld, empty=False) # this one will only add he additional map.
                    U.reset_env(env, reset_pose=RP.HOME_R, reset_pose_name='HOME_R', delay_scale_factor=1.5)
                    grasp_h_r = default_h_r # after using elevator, go back to the favorite position
                    name = default_name
                    # localize the robot
                    skill_name2obj[skill_name].localize_robot(floor=floor_num, bld=args.bld)
            elif skill_name == 'navigate_to_point_on_ground':
                is_success, reason_for_failure, history, return_info = \
                        skill_name2obj[skill_name].step(
                            **common_args,
                            execute=args.exec,
                            run_vlm=args.run_vlm,
                            debug=args.debug,
                            mask_image=selection_return_info['mask_image'], # avoids re-querying the gsam
                            bboxes=selection_return_info['bboxes'], # avoids re-querying the gsam
                        )
            else:
                raise ValueError(f"Unknown skill name: {skill_name}")

            history_i = update_history(is_success=is_success, reason_for_failure=reason_for_failure, history_i=history, args=args)
            # we update the skill_selection history to get better predictions everytime
            skill_histories['selection'][-1]['is_success'] = is_success
            skill_histories['selection'][-1]['env_reasoning'] = reason_for_failure
            skill_histories[skill_name].append(history_i)

            traj_save['skill_histories'] = skill_histories
            traj_save['task_query'] = task_query
            traj_save['total_skill_exec'] = select_ind + 1
            traj_save['selection_seq'] = selection_seq
            traj_save['cam_intr'].append(cam_intr)
            traj_save['cam_extr'].append(cam_extr)
            pickle.dump(traj_save, open(f'{log_dir}/eval_{eval_ind:03d}.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            print(colored("Trajectory step saved.", "blue"))

            if return_info['reset_required']:
                traj_success = False
                break

            # ask the user if the task is completed
            task_completed = U.confirm_user(True, 'Task completed? (y/n)', 'Task completion request.')
            if task_completed:
                print(colored(f"Run directory: {run_dir}", 'green'))
                print(colored(f"Task query: {task_query}", 'green'))
                traj_success = U.confirm_user(True, 'Success? (y/n)', 'Success.')
                break

        reason_for_failure = ""
        if not traj_success:
            print(colored("Please describe the reason for failure, example, bad depth image, execution failure, etc", "blue"))
            U.clear_input_buffer()
            reason_for_failure = input("Reason for failure: ")

        U.clear_input_buffer()
        print(colored("Enter number of subtasks completed", "red"))
        completed_subtasks = input("Number of subtasks: ")

        U.clear_input_buffer()
        print(colored("Enter total number of subtasks in the trial", "red"))
        total_subtasks = input("Total number of subtasks: ")

        U.clear_input_buffer()
        describe_traj = input("Describe the trajectory (use some reference or created trial id for future reference of this eval rollout:")
        logger_dict['success'].append(str(traj_success))
        logger_dict['reason_for_failure'].append(reason_for_failure)
        logger_dict['describe_traj'].append(describe_traj)
        # joint by : to make it easier to parse
        selection_seq = ':'.join(selection_seq)
        logger_dict['skill_selection_list'].append(selection_seq)
        logger_dict['eval_ind'].append(str(eval_ind))
        logger_dict['completed_subtasks'].append(completed_subtasks)
        logger_dict['total_subtasks'].append(total_subtasks)
        # save the logger in log_dir as a csv file
        U.save_csv(logger_dict, os.path.join(log_dir, 'eval.csv'))

        traj_save['skill_histories'] = skill_histories
        traj_save['task_query'] = task_query
        traj_save['traj_success'] = traj_success
        traj_save['total_skill_exec'] = select_ind+1
        pickle.dump(traj_save, open(f'{log_dir}/eval_{eval_ind:03d}.pkl', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    rospy.signal_shutdown("Shutdown")
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bld', type=str, default=None)
    parser.add_argument('--eval_id', type=int, help="eval_id=1 is the retrieving colored marker, eval_id=2 is the retrieving diet coke, eval_id=3 is rearranging chairs.")
    parser.add_argument('--skills', type=str, nargs='+', default=None, help='List of skills to add to the skill library. If None, all skills will be added.')
    parser.add_argument('--n_eval', type=int, default=1, help='Number of evaluations to run')
    parser.add_argument('--run_dir', type=str, default=None)
    parser.add_argument('--suffix', type=str, default=None)

    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--exec', action='store_true', help='execute the skill in the real robot.')
    parser.add_argument('--arm', type=str, default='left', choices=['right', 'left'])
    parser.add_argument('--run_vlm', action='store_true', help='run the vlm. Either run_vlm or oracle should be true.')
    parser.add_argument('--oracle', action='store_true', help='use human (not set up for all skills) for the skill execution')
    parser.add_argument('--method', default="ours", choices=["ours", "llm_baseline", "ours_no_markers"])

    parser.add_argument('--n_skill_selection', type=int, default=30, help='Maximum number of skill selection beyond which the trial will be stopped.')
    parser.add_argument('--add_selection_history', action='store_true', help='Add short-term memory')
    parser.add_argument('--add_past', action='store_true', help='Add long-term memory.')
    parser.add_argument('--floor_num', type=int, help='starting floor number', default=None)
    args = parser.parse_args()
    if args.skills is None:
        args.skills = ['pick_up_object', 'move', 'goto_landmark', 'open_door', 'call_elevator', 'use_elevator', 'push_obs_gr', 'navigate_to_point_gr']
    assert args.run_vlm or args.oracle
    assert args.eval_id is not None
    assert args.floor_num is not None, "Please add the correct --floor_num flag."
    assert args.bld is not None, "Please add the correct --bld flag."
    assert args.run_dir is not None, "Please add the correct --run_dir flag."
    main(args)
