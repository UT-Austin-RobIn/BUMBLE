import os, sys
import time

# sys.path.append(os.path.join(os.getcwd(),"../third_party/Grounded-Segment-Anything","GroundingDINO"))


# If you have multiple GPUs, you can set the GPU to use here.
# The default is to use the first GPU, which is usually GPU 0.

import argparse
import os
import copy

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict

import supervision as sv

# segment anything
from segment_anything import build_sam, SamPredictor, build_sam_hq_vit_l, build_sam_hq_vit_h, build_sam_hq_vit_b
import cv2
import numpy as np
import matplotlib.pyplot as plt


from huggingface_hub import hf_hub_download
import groundingdino.datasets.transforms as T

from bumble.utils.utils import get_palette


class GroundedSamWrapper:
    def __init__(self, sam_ckpt_path, load_from_hf=True):
        # Use this command for evaluate the Grounding DINO model
        # Or you can download the model by yourself
        ckpt_repo_id = "ShilongLiu/GroundingDINO"
        ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
        ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"

        if load_from_hf:
            self.groundingdino_model = self.load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
        else:
            self.groundingdino_model = self.load_groundingdino_model()

        device = 'cpu'
        sam = None
        if 'sam_vit_h_4b8939.pth' in sam_ckpt_path:
            sam = build_sam(checkpoint=sam_ckpt_path)
        elif 'sam_hq_vit_l.pth' in sam_ckpt_path:
            sam = build_sam_hq_vit_l(checkpoint=sam_ckpt_path)
        elif 'sam_hq_vit_h.pth' in sam_ckpt_path:
            sam = build_sam_hq_vit_h(checkpoint=sam_ckpt_path)
        elif 'sam_hq_vit_b.pth' in sam_ckpt_path:
            sam = build_sam_hq_vit_b(checkpoint=sam_ckpt_path)
            device = 'cuda'
        else:
            raise ValueError("Unknown SAM model checkpoint")
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    def load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cuda'):
        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

        args = SLConfig.fromfile(cache_config_file)
        model = build_model(args)
        args.device = device

        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location='cuda')
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        print("Model loaded from {} \n => {}".format(cache_file, log))
        _ = model.eval()
        return model

    def load_groundingdino_model(self):
        def load_model(model_config_path, model_checkpoint_path, device):
            args = SLConfig.fromfile(model_config_path)
            args.device = device
            model = build_model(args)
            checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
            load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
            print(load_res)
            _ = model.eval()
            return model
        config_file = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        grounded_checkpoint = "groundingdino_swint_ogc.pth"
        model = load_model(config_file, grounded_checkpoint, device="cuda")
        return model

    def transform(self, image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.fromarray(image).convert("RGB")
        image_transformed, _ = transform(image_source, None)
        return image, image_transformed

    def split_connected_components(self, masks):
        mask_split = []
        for i in range(masks.shape[0]):
            total_connected_components, mask = cv2.connectedComponents(masks[i][0].astype(np.uint8))
            uniq_values = np.unique(mask)
            for value in uniq_values[1:]:
                mask_split.append(mask==value)
        mask_split = np.array(mask_split).astype(np.uint8)
        mask_split = np.expand_dims(mask_split, axis=1)
        return mask_split

    def combine_masks(self, masks):
        final_mask = np.zeros((masks.shape[2], masks.shape[3]), dtype=np.uint8)
        masks = masks.cpu().detach().numpy()
        masks = self.split_connected_components(masks)

        # erosion and dilation to remove noise
        for i in range(masks.shape[0]):
            # erode and then dilate to remove noise
            masks[i][0] = cv2.erode(masks[i][0].astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
            masks[i][0] = cv2.dilate(masks[i][0].astype(np.uint8), np.ones((3, 3), np.uint8), iterations=3)
        final_mask = final_mask + masks[0][0]

        for i in range(1, masks.shape[0]):
            final_mask[masks[i][0] > 0] = i + 1

        return final_mask

    def segment(self, image_np, prompts, box_threshold=0.3, text_threshold=0.25, filter_threshold=200, mask_input=None):
        image_source, image = self.transform(image_np)
        prompt_text = ""
        for prompt in prompts:
            prompt_text += (prompt + ".")

        boxes, logits, phrases = predict(
            model=self.groundingdino_model,
            image=image,
            caption=prompt_text,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )
        print(boxes, logits, phrases)

        if boxes.shape[0] == 0:
            return np.array([])

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB

        # box: normalized box xywh -> unnormalized xyxy
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        device = "cuda"
        # set image
        self.sam_predictor.set_image(image_source)
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)
        if mask_input is not None:
            # convert to torch tensor
            mask_input = torch.from_numpy(mask_input).float().to(device)
        masks, _, _ = self.sam_predictor.predict_torch(
                    point_coords = None,
                    point_labels = None,
                    boxes = transformed_boxes,
                    mask_input = mask_input,
                    multimask_output = False,
                )

        intermediate_final_mask = self.combine_masks(masks)

        filter_indices = []
        for i in range(1, intermediate_final_mask.max() + 1):
            if np.sum(intermediate_final_mask == i) < filter_threshold:
                filter_indices.append(i)

        final_mask = np.zeros_like(intermediate_final_mask)
        count = 0
        for i in range(1, intermediate_final_mask.max() + 1):
            if i not in filter_indices:
                final_mask[intermediate_final_mask == i] = count + 1
                count += 1

        mask_image_pil = Image.fromarray(final_mask) # .convert("RGBA")
        mask_image_pil.putpalette(get_palette())
        return mask_image_pil
