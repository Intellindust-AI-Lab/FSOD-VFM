import os
import torch
import pycocotools.coco
import PIL.Image
import support_util
import cv2
from tqdm import tqdm
import numpy as np

def load_voc2007_coco_json(json_path, images_root):
    coco_style_loader = pycocotools.coco.COCO(json_path)
    img_ids = coco_style_loader.getImgIds()
    img_info_list = coco_style_loader.loadImgs(img_ids)
    image_paths = [os.path.join(images_root, img_info['file_name']) for img_info in img_info_list]
    return image_paths, coco_style_loader


def get_candidate_masks(sam2_mask_generator, img_pil, device='cpu'):
    
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        masks = sam2_mask_generator.generate(np.array(img_pil))
    candidate_masks = [torch.from_numpy(m['segmentation'].astype(np.float32)).unsqueeze(0).to(device)  for m in masks]
    candidate_box = [list(map(int, m['bbox'])) for m in masks]
    predicted_iou = [m['predicted_iou'] for m in masks]
    
    return torch.stack(candidate_masks, dim=0), candidate_box, predicted_iou


def get_masks_consistency_score_batch(sam2_predictor, pil_img, ref_boxes, device="cuda"):
    """
    Apply SAM on a batch of bounding boxes over one image.

    Args:
        sam2_predictor: Initialized SAM predictor.
        pil_img: PIL Image object.
        ref_boxes: list or numpy array of shape (N, 4), each [x1, y1, x2, y2].
        device: CUDA or CPU.

    Returns:
        masks: list of binary numpy arrays (H x W).
        scores: list of float scores (mask_area / bbox_area).
    """
    # Convert boxes to torch.Tensor
    ref_boxes = torch.tensor(ref_boxes, dtype=torch.bfloat16, device=device)
    ref_boxes[:, 2] = ref_boxes[:, 0] + ref_boxes[:, 2]
    ref_boxes[:, 3] = ref_boxes[:, 1] + ref_boxes[:, 3]
    
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        sam2_predictor.set_image(np.array(pil_img))
        masks, score, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=ref_boxes,              # shape: (N, 4)
            multimask_output=False,     # one mask per box
        )

    return torch.from_numpy(masks).to(device), score


def mask_to_bbox_xyxy_cv2(mask_np, min_area=None, max_area=None):
    """
    Converts a binary mask to its tightest bounding box in [x_min, y_min, x_max, y_max] format using OpenCV.
    Returns None if the mask is empty.
    """
    if mask_np.sum() == 0:
        return None

    mask_uint8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None # No contours found (e.g., very small or scattered pixels)
    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)
    area = w * h

    if min_area is not None:
        if area < min_area:
            return None
    
    if max_area is not None:
        if area > max_area:
            return None

    x_min, y_min = x, y
    x_max, y_max = x + w, y + h

    return [x_min, y_min, x_max, y_max]