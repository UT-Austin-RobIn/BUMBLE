import os
import sys
import cv2
import csv
import select
from PIL import Image
import numpy as np
import random
import time
from easydict import EasyDict

import plotly.graph_objects as go
import plotly.express as px

# Plottgin gpt4 chats
import re
import json
import base64
import io
import textwrap
import matplotlib.pyplot as plt
import psutil
import signal
from termcolor import colored
import yaml
import bumble.tiago.RESET_POSES as RP
import bumble.tiago.prompters.vlms as vlms # GPT4V

def extract_json(response, key):
    json_part = re.search(r"\{.*\}", response, re.DOTALL)
    parsed_json = {}
    if json_part:
        json_data = json_part.group()
        # Parse the JSON data
        parsed_json = json.loads(json_data)
    else:
        print("No JSON data found ******\n", response)
    return parsed_json[key]

def get_model(model_name='gpt-4o-2024-05-13'):
    vlm = None
    if 'gpt' in model_name:
        vlm = vlms.GPT4V(openai_api_key=os.environ['OPENAI_API_KEY'], model_name=model_name)
    if 'claude' in model_name:
        vlm = vlms.Ant(openai_api_key=os.environ['ANTHROPIC_API_KEY'], model_name=model_name)
    return vlm

def save_yaml(cfg, path):
    with open(path, 'w') as f:
        yaml.dump(cfg, f)
    return

def _user_input(text, valid_inputs):
    _input = input(text)
    while _input not in valid_inputs:
        _input = input(text)
    return _input

# Function to clear input buffer
def clear_input_buffer():
    while select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.read(1)

def check_if_already_reset(env, reset_pose, reset_arms=True):
    already_reset = True
    needs_reset = []
    for key, value in reset_pose.items():
        if (key == 'torso') and (value is not None):
            already_reset = env.tiago.torso.is_at_joint(value)
            if not already_reset:
                needs_reset.append('torso')
        elif (key == 'left') and reset_arms and (value is not None):
            already_reset = env.tiago.arms['left'].is_at_joint(value[:-1])
            if not already_reset:
                needs_reset.append('left')
        elif (key == 'right') and reset_arms and (value is not None):
            already_reset = env.tiago.arms['right'].is_at_joint(value[:-1])
            if not already_reset:
                needs_reset.append('right')
    already_reset = already_reset and (len(needs_reset) == 0)
    return already_reset, needs_reset

def confirm_user(execute, question, info_string=None):
    clear_input_buffer()
    if info_string is not None:
        print(colored(info_string, 'magenta'))
    if not execute:
        return execute
    _input = _user_input(question, valid_inputs=['y', 'n'])
    return _input == 'y'

def reset_env(
        env,
        reset_pose,
        reset_pose_name=None,
        reset_arms=True,
        int_pose=None,
        int_pose_name=None,
        wait_user=True,
        delay_scale_factor=6.0
    ):
    print(colored(f"Moving to {reset_pose_name} pose", 'magenta'))
    print(colored(f"Resetting arms: {reset_arms}", 'magenta'))
    print(colored(f"Resetting pose: {reset_pose}", 'magenta'))
    skip_reset = False
    already_reset, list_needs_reset = check_if_already_reset(env, reset_pose, reset_arms)
    print(colored(f"Needs reset: {list_needs_reset}", 'magenta'))
    print(colored(f"Already reset: {already_reset}", 'magenta'))
    if already_reset:
        print(colored(f"Already at {reset_pose_name} pose", 'magenta'))
        skip_reset = True
    if ('left' not in list_needs_reset) and ('right' not in list_needs_reset):
        int_pose = None
        reset_pose['left'] = None
        reset_pose['right'] = None
    if ('left' not in list_needs_reset):
        if int_pose is not None:
            int_pose['left'] = None
        reset_pose['left'] = None
    if ('right' not in list_needs_reset):
        if int_pose is not None:
            int_pose['right'] = None
        reset_pose['right'] = None
    if int_pose is not None:
        print("FIRST WE ARE GOING TO MOVE TO INTERMEDIATE POSE")
        print(colored(f"Moving to {int_pose_name} pose", 'magenta'))
        print(colored(f"Resetting arms: {reset_arms}", 'magenta'))
        print(colored(f"Resetting pose: {int_pose}", 'magenta'))

    user_input = True
    if not skip_reset:
        user_input = confirm_user(True, 'Move to reset pose? (y/n): ') # if the user says no, then we will not move to the reset pose. Simply return False
    if not user_input:
        return False
    if int_pose is not None:
        env.reset(reset_arms=reset_arms, reset_pose=int_pose, allowed_delay_scale=6.0, delay_scale_factor=delay_scale_factor, skip_reset=skip_reset, wait_user=False)
    env.reset(reset_arms=reset_arms, reset_pose=reset_pose, allowed_delay_scale=6.0, delay_scale_factor=delay_scale_factor, skip_reset=skip_reset, wait_user=wait_user)
    return True

def save_model_output(
    rgb,
    response,
    subtitles,
    img_file,
):
    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    ax[0].imshow(rgb); ax[0].axis('off')
    if len(subtitles) > 0:
        ax[0].set_title(subtitles[0])
    import textwrap
    text = response
    textwrap = textwrap.TextWrapper(width=75)
    text = textwrap.fill(text)
    ax[1].imshow(np.ones_like(rgb) * 255); ax[1].axis('off')
    ax[1].text(0, 0, text, fontsize=10, color='black', wrap=True)
    if len(subtitles) > 1:
        ax[1].set_title(subtitles[1])
    ax[1].set_xlim(0, rgb.shape[1])
    ax[1].set_ylim(0, rgb.shape[0])
    plt.savefig(img_file)
    plt.clf()
    return

def kill_all_child_processes():
    """
    Kill all child processes of the current process.
    """
    # Get the current process ID
    current_pid = os.getpid()
    # Get the process object for the current process
    current_process = psutil.Process(current_pid)
    # Get the child processes of the current process
    child_processes = current_process.children(recursive=True)
    # Kill each child process
    for child in child_processes:
        os.kill(child.pid, signal.SIGTERM)
    # wait for the child processes to terminate
    _, _ = psutil.wait_procs(child_processes, timeout=5)
    return

def save_csv(log, path):
    # log is easydict object. save it as csv
    # each key has a list of values. Each key is a column and each value is a row
    with open(path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(log.keys())
        writer.writerows(zip(*log.values()))

def load_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    log = EasyDict()
    keys = data[0]
    for key in keys:
        log[key] = []
    for row in data[1:]:
        for i, key in enumerate(keys):
            log[key].append(row[i])
    return log

def save_image(image, path):
    """
    Save an image to a file.
    Args:
        image (np.ndarray): The image to save.
        path (str): The path to save the image to.
    """
    # Convert the image to the correct format
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # Save the image
    cv2.imwrite(path, image)
    return

def encode_image(image):
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', bgr)
    return buffer

def decode_image(buffer):
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_user_input(rgb, num_pts=1):
    print("Click on the image to select the point")
    def onclick(event):
        # Ignore clicks outside the axes
        if event.inaxes is not None:
            # Convert the coordinates to integers and add to the list
            ix, iy = int(event.xdata), int(event.ydata)
            clicked_points.append((ix, iy))
            print(f"Point added: x = {ix}, y = {iy}")

    plt.ion()
    fig, ax = plt.subplots()
    ax.imshow(rgb, origin='upper')

    # List to store the points
    clicked_points = []

    # Connect the click event handler
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    while len(clicked_points) < 1:
        plt.pause(0.1)  # Pause to yield control to the GUI event loop
    # wait for the user to click on the image
    plt.ioff()
    plt.close()

    return clicked_points

def get_bbox_center_dist(bbox_list, robot_pos, pcd):
    '''
        bbox_list: list of bboxes in the format [id, x1, y1, x2, y2]
        robot_pos: robot position in the format [x, y, z]
        pcd: point cloud data in the format [x, y, z]
        Returns: list of distances of the centers of the bboxes from the robot position
    '''
    bbox_id2dist = {}
    center_pts = [(bbox[0], (bbox[1]+bbox[3])//2, (bbox[2]+bbox[4])//2) \
            for bbox in bbox_list]
    for bbox_id, x, y in center_pts:
        dist = np.linalg.norm(pcd[y, x, :2] - robot_pos[:2])
        bbox_id2dist[bbox_id] = dist
    return bbox_id2dist

def convert_segsemantic_to_bboxes(seg_semantic):
    '''
        seg_semantic: Image of shape (H, W) with pixel values as object ids.
        Returns: List of bboxes of the form [obj_id, xmin, ymin, xmax, ymax]
    '''
    obj_ids = np.unique(seg_semantic)
    bboxes = []
    for obj_id in obj_ids:
        mask = seg_semantic == obj_id
        ys, xs = np.where(mask)
        xmin, ymin, xmax, ymax = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        bboxes.append([obj_id, xmin, ymin, xmax, ymax])
    return bboxes

def add_bbox_to_rgb(rgb, bboxes, ignore_ids=[], keep_ids=None, color=(255, 255, 255)):
    '''
        rgb: np.ndarray of shape (H, W, 3)
        bboxes: list of bboxes in the format [id, x1, y1, x2, y2, other stuff]
        ignore_ids: list of ids to ignore
        color: color of the bbox in BGR format
    '''
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for i, bbx in enumerate(bboxes):
        if bbx[0] not in ignore_ids:
            if (keep_ids is None) or (bbx[0] in keep_ids):
                rgb = cv2.rectangle(rgb, (bbx[1], bbx[2]), (bbx[3], bbx[4]), color, 2)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb

def create_border_mask(mask):
    # Convert the boolean mask to a binary image format expected by OpenCV
    binary_mask = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"Number of contours: {len(contours)}")
    if len(contours) == 0:
        return mask
    # Create an empty mask to draw the contours on
    border_mask = np.zeros_like(binary_mask)
    # Draw the contours - this marks the borders
    cv2.drawContours(border_mask, contours, -1, (255), thickness=3)
    # Convert the border mask back to a boolean format
    border_mask_bool = border_mask.astype(bool)
    return border_mask_bool

def remove_nan_pcd(pcd, rgb=None):
    # Remove any NaN values from the point cloud data
    mask = np.isnan(pcd).any(axis=-1)
    pcd = pcd[~mask]
    if rgb is not None:
        rgb = rgb[~mask]
    return pcd, rgb


def plotly_draw_3d_pcd(pcd_points, pcd_colors=None, addition_points=None, marker_size=3, equal_axis=True, title="", offline=False, no_background=False, default_rgb_str="(255,0,0)",additional_point_draw_lines=False, uniform_color=False):

    if pcd_colors is None:
        color_str = [f'rgb{default_rgb_str}' for _ in range(pcd_points.shape[0])]
    else:
        color_str = ['rgb('+str(r)+','+str(g)+','+str(b)+')' for r,g,b in pcd_colors]

    # Extract x, y, and z columns from the point cloud
    x_vals = pcd_points[:, 0]
    y_vals = pcd_points[:, 1]
    z_vals = pcd_points[:, 2]

    # Create the scatter3d plot
    rgbd_scatter = go.Scatter3d(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        mode='markers',
        marker=dict(size=3, color=color_str, opacity=0.8)
    )
    data = [rgbd_scatter]
    if addition_points is not None:
        assert(addition_points.shape[-1] == 3)
        # check if addition_points are three dimensional
        if len(addition_points.shape) == 2:
            addition_points = [addition_points]
        for points in addition_points:
            x = points[:, 0]
            y = points[:, 1]
            z = points[:, 2]
            if additional_point_draw_lines:
                mode = "lines+markers"
            else:
                mode = "markers"
            marker_dict = dict(size=marker_size,
                                opacity=0.8)

            if uniform_color:
                marker_dict["color"] = f'rgb{default_rgb_str}'
            rgbd_scatter2 = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode=mode,
                marker=marker_dict,
                )
            data.append(rgbd_scatter2)

    if equal_axis:
        scene_dict = dict(
            aspectmode='data',
        )
    else:
        scene_dict = dict()
    # Set the layout for the plot
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        # axes range
        scene=scene_dict,
        title=dict(text=title, automargin=True)
    )

    fig = go.Figure(data=data, layout=layout)

    if no_background:
        fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
            yaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
            zaxis=dict(showbackground=False, zeroline=False, showgrid=False, showticklabels=False, showaxeslabels=False, visible=False),
        ),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        margin=dict(l=0, r=0, b=0, t=0),  # No margins
        showlegend=False,
    )


    if not offline:
        fig.show()
    else:
        return fig

def wrap_text_preserving_newlines(text, width):
    # Split the text by newline characters to preserve them
    lines = text.split('\n')

    # Wrap each line individually and join them back with newline characters
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines, preserving original newlines
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def save_video(imgs, video_path, duration=-1, fps=60):
    """
    Save a video from a list of images
    :param imgs: list of images
    :param video_path: path to save the video
    :param duration: duration of the video in seconds
    :param fps: frames per second
    :return:
    """
    if duration > 0:
        if len(imgs) < duration * fps:
            for _ in range(duration * fps - len(imgs)):
                imgs.append(imgs[-1])
        elif len(imgs) > duration * fps:
            imgs = imgs[:duration * fps]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(video_path, fourcc, fps, (imgs[0].shape[1], imgs[0].shape[0]))
    for i in range(len(imgs)):
        img = cv2.cvtColor(imgs[i], cv2.COLOR_RGB2BGR)
        out.write(img)
    out.release()
    return

def add_arrows_on_image(
        image: np.ndarray,
        center_pt: tuple[int, int],
        directions: list[str],
        bgr_color=(255, 0, 0),
        length=20,
        thickness=2,
    ):
        # convert pts to int
        center_pt = (int(center_pt[0]), int(center_pt[1]))
        rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for direction in directions:
            if direction == 'up':
                cv2.arrowedLine(rgb, (center_pt[1], center_pt[0]), (center_pt[1], center_pt[0]-length), bgr_color, thickness)
            elif direction == 'down':
                cv2.arrowedLine(rgb, (center_pt[1], center_pt[0]), (center_pt[1], center_pt[0]+length), bgr_color, thickness)
            elif direction == 'left':
                cv2.arrowedLine(rgb, (center_pt[1], center_pt[0]), (center_pt[1]-length, center_pt[0]), bgr_color, thickness)
            elif direction == 'right':
                cv2.arrowedLine(rgb, (center_pt[1], center_pt[0]), (center_pt[1]+length, center_pt[0]), bgr_color, thickness)
            elif direction == 'clockwise':
                # show an arrow going inside the image
                # probably at the angle of 45 degrees
                raise NotImplementedError
                cv2.arrowedLine(rgb, (center_pt[1], center_pt[0]), (center_pt[1]+length, center_pt[0]-length), bgr_color, thickness)
            elif direction == 'anticlockwise':
                raise NotImplementedError

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb

def wrap_text(text, max_width):
    """Wrap text to the specified width."""
    words = text.split()
    wrapped_lines = []
    current_line = []

    for word in words:
        # Check if adding the next word would exceed the max width
        if sum(len(w) for w in current_line) + len(word) + len(current_line) > max_width:
            wrapped_lines.append(' '.join(current_line))
            current_line = [word]
        else:
            current_line.append(word)

    # Add the last line
    wrapped_lines.append(' '.join(current_line))

    return '\n'.join(wrapped_lines)


def convert_str_to_color(str_color: str) -> tuple:
    """
    Args:
        str_color (str): string color in the format of 'r,g,b'

    Returns:
        tuple: tuple of int
    """
    color = ()
    if str_color == 'white':
        color = (255, 255, 255)
    elif str_color == 'black':
        color = (0, 0, 0)
    elif str_color == 'dark_gray':
        color = (220, 220, 220)
    elif str_color == 'gray':
        color = (128, 128, 128)
    elif str_color == 'red':
        color = (255, 0, 0)
    elif str_color == 'green':
        color = (0, 255, 0)
    elif str_color == 'blue':
        color = (0, 0, 255)
    return color

def convert_img_path2segm_path(img_path: str, ooi: list[str]) -> str:
    assert isinstance(ooi, list), f"ooi should be a list, but got {type(ooi)}"
    segm_path = img_path.replace('img', 'segm')
    ext = os.path.splitext(segm_path)[-1]
    ooi_name = ','.join([ooi_name.replace(' ', '_') for ooi_name in ooi])
    segm_path = segm_path.replace(ext, f'_{ooi_name}.png')
    return segm_path

def overlay_xmem_mask_on_image(rgb_img, mask, use_white_bg=False, rgb_alpha=0.7):
    """
    Args:
        rgb_img (np.ndarray):rgb images
        mask (np.ndarray)): binary mask
        use_white_bg (bool, optional): Use white backgrounds to visualize overlap. Note that we assume mask ids 0 as the backgrounds. Otherwise the visualization might be screws up. . Defaults to False.

    Returns:
        np.ndarray: overlay image of rgb_img and mask
    """
    colored_mask = Image.fromarray(mask)
    colored_mask.putpalette(get_palette())
    colored_mask = np.array(colored_mask.convert("RGB"))
    if use_white_bg:
        colored_mask[mask == 0] = [255, 255, 255]
    overlay_img = cv2.addWeighted(rgb_img, rgb_alpha, colored_mask, 1-rgb_alpha, 0)

    return overlay_img

def get_palette(palette="davis"):
    davis_palette = b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0'
    youtube_palette = b'\x00\x00\x00\xec_g\xf9\x91W\xfa\xc8c\x99\xc7\x94b\xb3\xb2f\x99\xcc\xc5\x94\xc5\xabyg\xff\xff\xffes~\x0b\x0b\x0b\x0c\x0c\x0c\r\r\r\x0e\x0e\x0e\x0f\x0f\x0f'
    if palette == "davis":
        return davis_palette
    elif palette == "youtube":
        return youtube_palette

def plot_gpt_chats(chats, save_key, save_dir, size=(10, 30), font_size=10):
    '''
        Chats is a list of conversations with gpt-4. We want to plot it as an image.
        All the messages with role == system will be in red otherwise green.
        Plot the images in the message as well as the text.
    '''
    colors = ['red', 'green']
    # each chat is an image
    for index, chat in enumerate(chats):
        # make a figure with variable number of subplots
        fig, ax = plt.subplots(len(chat)+1, 1)
        # remove subplot borders
        for ax_ in ax:
            ax_.axis('off')
        fig.set_size_inches(*size)
        # each message is a row
        for fig_row, message in enumerate(chat):
            '''
            if fig_row == len(chat)-1:
                wrapped_text = textwrap.fill("######## Actual GPT-4 Response below [this text was not part of the conversation] ########", width=100)
                ax[fig_row].text(
                        0.5, 0.5, wrapped_text, wrap=True, \
                        horizontalalignment='center', verticalalignment='center', color='black', \
                        fontsize=font_size, bbox=dict(facecolor='white', edgecolor='white', alpha=0.5))
                fig_row += 1
            '''
            # Adjust the ax to the size of the content
            # print plotting role and content type
            print(message['role'], message['content'][0]['type'])
            if (message['role'] == 'system') or (message['role'] == 'assistant'):
                color = colors[0]
            else:
                color = colors[1]
            # each message is a row
            # Iterate over the content and plot them in a row
            # Create columns for each content in ax[fig_row]
            imgs = []
            for fig_col, content in enumerate(message['content']):
                if content['type'] == 'text':
                    # wrap the text within the box
                    print(content['text'])
                    # wrapped_text = wrap_text_preserving_newlines(content['text'], width=100)
                    wrapped_text = textwrap.fill(content['text'], width=100)
                    # plot the text
                    ax[fig_row].text(
                            0.5, 0.5, wrapped_text, wrap=True, \
                            horizontalalignment='center', verticalalignment='center', color=color, \
                            fontsize=font_size, bbox=dict(facecolor='white', edgecolor='white', pad=0))
                elif content['type'] == 'image_url':
                    img_list = content['image_url']["url"]
                    base64_encoded_image_list = img_list.split(",")[1:]  # Remove the 'data:image/jpeg;base64,' prefix

                    for base64_encoded_image in base64_encoded_image_list:
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(io.BytesIO(image_data))
                        image_array = np.array(image)
                        imgs.append(image_array)
            if len(imgs) > 0:
                # if greater than 4 images, then sample 4 images with first and last image in the row
                if len(imgs) > 4:
                    imgs_first, imgs_last = imgs[0], imgs[-1]
                    imgs = random.sample(imgs[1:-1], 2)
                    imgs = [imgs_first] + imgs + [imgs_last]

                # plot the images
                # joint all the images in the row
                image_array = np.concatenate(imgs, axis=1)
                ax[fig_row].imshow(image_array)

        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, f'chat_{index}_{save_key}.pdf'))
        plt.close()
        plt.clf()
    return
