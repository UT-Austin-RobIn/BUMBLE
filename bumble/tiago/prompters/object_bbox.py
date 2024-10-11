import os
import cv2
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import bumble.utils.utils as U
import bumble.tiago.prompters.vlms as vlms # GPT4V

class ObjectBbox:
    bbox: list
    obj_id: str
    obj_name: str
    env_id: int
    dist2robot: float

def bbox_prompt_img(im, bboxes, prompt_args, info):
    '''
    im: np.ndarray (H, W, 3)
    bboxes: list of tuples (env_id, x1, y1, x2, y2)
    Example of prompt_args:
        prompt_args = {
            "color": (0, 0, 0),
            "mix_alpha": 0.6,
            'radius': int(img_size * radius_per_pixel),
            'thickness': 2,
            'fontsize': int(img_size * 30 * radius_per_pixel),
            'rgb_scale': 255,
            'add_object_boundary': False,
            'add_dist_info': False, # not used in this function
            'add_arrows_for_path': False,
            'path_start_pt': (0, 0),
            'path_end_pt': (0, 0),
            'radius_per_pixel': 0.03,
            'plot_outside_bbox': False,
        }
    info:
        bbox_ignore_ids: list of int
        bbox_id2dist: dict of int to float
    '''
    overlay = im.copy()
    white = (
        prompt_args['rgb_scale'],
        prompt_args['rgb_scale'],
        prompt_args['rgb_scale'],
    )
    text_radius = prompt_args['radius'] / 3**0.5
    char_len_adjust = 1.0
    center_adjust = 1.0
    obj_bbox_list = []
    label_list = []
    # label_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N'] if 'label_list' not in prompt_args.keys() else prompt_args['label_list']
    if 'label_list' not in prompt_args.keys():
        # add all 26 alphabets
        label_list = [chr(i) for i in range(65, 91)]
    else:
        label_list = prompt_args['label_list']
    count_ind = 0
    for ind, bbox in enumerate(bboxes):
        env_id, x1, y1, x2, y2 = bbox
        if env_id in info['bbox_ignore_ids']:
            continue

        count_ind += 1
        obj_bbox = ObjectBbox()
        obj_bbox.bbox = bbox
        # obj_bbox.obj_id = str(count_ind) # this ensures that the object id are sequential
        # assign a character to the object id starting with 'a'
        # obj_bbox.obj_id = chr(ord('a') + count_ind - 1).upper()
        obj_bbox.obj_id = label_list[count_ind - 1]
        obj_bbox.env_id = env_id
        obj_bbox.dist2robot = info['bbox_id2dist'][env_id]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        point = None
        if prompt_args['add_object_boundary']:
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                prompt_args['color'],
                prompt_args['thickness'],
            )
            # add a label to the overlay image with a circle and text in it
            # point = int(x1 + prompt_args['radius']), int(y1 + prompt_args['radius'])
            point = int(x1 - prompt_args['radius']), int(y1 - prompt_args['radius'])
            # cap the point to be within the image with the radius
            point = (
                max(min(point[0], im.shape[1] - prompt_args['radius']), prompt_args['radius']),
                max(min(point[1], im.shape[0] - prompt_args['radius']), prompt_args['radius']),
            )
        else:
            # if area is small, put the label outside the bbox
            # if ((x2 - x1) * (y2 - y1))/(im.shape[0] * im.shape[1]) < 0.05:
            #     point = (x1 - prompt_args['radius'], y1 - prompt_args['radius'])
            # else:
            if ('plot_outside_bbox' in prompt_args.keys()) and prompt_args['plot_outside_bbox']:
                # move the x-coordinate to the left of the bbox and y as the mean of y1 and y2
                point = (x1 - prompt_args['radius'], int((y1 + y2) / 2))
            else:
                point = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            point = (
                max(min(point[0], im.shape[1] - prompt_args['radius']), prompt_args['radius']),
                max(min(point[1], im.shape[0] - prompt_args['radius']), prompt_args['radius']),
            )
        cv2.circle(
            overlay,
            point,
            prompt_args['radius'],
            prompt_args['color'],
            thickness=-1,
        )
        cv2.circle(overlay, point, prompt_args['radius'], white, -1)
        fontsize = prompt_args['fontsize']/im.shape[0]
        cv2.putText(
            overlay,
            obj_bbox.obj_id,
            (int(point[0]-(text_radius/center_adjust)),int(point[1]+text_radius)),
            cv2.FONT_HERSHEY_SIMPLEX,
            char_len_adjust*fontsize, prompt_args['color'], prompt_args['thickness'],
            cv2.LINE_AA,
        )
        obj_bbox_list.append(obj_bbox)
    if ('add_arrows_for_path' in prompt_args.keys()) and prompt_args['add_arrows_for_path']:
        start_pt = info['path_start_pt']
        end_pt = info['path_end_pt']
        start_pt = (int(start_pt[0]), int(start_pt[1]))
        end_pt = (int(end_pt[0]), int(end_pt[1]))
        cv2.arrowedLine(
            overlay,
            start_pt,
            end_pt,
            (255, 0, 0), # red color,
            prompt_args['thickness'],
        )
    im = cv2.addWeighted(
        overlay, prompt_args['mix_alpha'],
        im, 1 - prompt_args['mix_alpha'],
        0,
    )

    return im, obj_bbox_list
