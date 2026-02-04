import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2 
import torch.nn.functional as F
import pickle

def extract_masks_from_support(sam_predictor, pil_img, ref_boxes, device="cuda"):
    """
    Use SAM to extract masks from a list of bounding boxes on a support image.
    
    Args:
        sam_predictor: Initialized SAM predictor (e.g., SAM2 predictor).
        img: numpy image (H x W x 3), loaded with cv2 or PIL.
        img_path: str, image file path.
        ref_boxes: a bbox with [x1, y1, x2, y2] 
        device: torch device string.

    Returns:
        reference_data: dict with masks for the given class.
    """
    
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        sam_predictor.set_image(pil_img)
        masks, scores, _ = sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=torch.tensor(ref_boxes)[None, :],  # ensure shape (1, 4)
            multimask_output=False,
        )
      
    return masks[0]
    
def get_dinov2_features(dinov2_model, dinov2_transform, pil_img, device='cpu'):
    pil_img = resize_with_aspect_ratio(pil_img, target_long_side=630, patch_size=14)
    image_tensor = dinov2_transform(pil_img)[None].to(device)
    with torch.inference_mode():
        output = dinov2_model.get_intermediate_layers(image_tensor, n =1, reshape=True, return_class_token=True, norm=False)
        output = torch.stack([out[0] for out in output], dim=0).sum(dim=0)
        return output # Shape: (B, C, H_feat, W_feat)

def resize_with_aspect_ratio(img_pil, target_long_side=1024, patch_size=16):
    """
    Resize a PIL image to have a specific long side, maintaining aspect ratio,
    and ensure new dimensions are multiples of the patch size.
    Uses BICUBIC filter for resampling.

    Args:
        img_pil (PIL.Image): Input image.
        target_long_side (int): Desired size of the longer side.
        patch_size (int): Size of the patches, new dimensions must be multiples of this.

    Returns:
        PIL.Image: Resized image with dimensions as multiples of patch_size.
    """
    orig_width, orig_height = img_pil.size
    aspect_ratio = orig_width / orig_height

    # Calculate initial resized dimensions based on long side
    if orig_width >= orig_height:
        new_width = target_long_side
        new_height = int(target_long_side / aspect_ratio)
    else:
        new_height = target_long_side
        new_width = int(target_long_side * aspect_ratio)

    # Ensure dimensions are multiples of patch_size
    # Using floor division to guarantee we don't exceed target_long_side
    new_width = max((new_width // patch_size), 1) * patch_size
    new_height = max((new_height // patch_size),1) * patch_size

    return img_pil.resize((new_width, new_height), resample=Image.BICUBIC)

def resize_mask_to_features(mask_np, feature_map_shape):
    H_feat, W_feat = feature_map_shape[0], feature_map_shape[1]
    
    # Handle different input dimensions
    if mask_np.ndim == 3:
        # If input is 3D (e.g., batch dimension), take the first mask
        if mask_np.shape[0] == 1:
            mask_np = mask_np[0]  # Remove batch dimension
        else:
            # If multiple masks, take the first one
            mask_np = mask_np[0]
    
    # Ensure mask is 2D
    if mask_np.ndim != 2:
        raise ValueError(f"Expected 2D mask, got {mask_np.ndim}D with shape {mask_np.shape}")
    
    # cv2.resize expects dsize as (width, height), not (height, width)
    resized_mask = cv2.resize(mask_np.astype(np.float32), dsize=(W_feat, H_feat))
    return (resized_mask > 0.5).astype(np.float32)
    
def extract_support_features(support_data, sam2_predictor, feat_extractor_name, feat_extractor, image_transform, data_dir, device='cpu'):
    '''
    support_data: dict[class_name] = list of dict with keys:
        - 'image': image path (relative to data_dir)
        - 'bbox': list of one or more [x1, y1, x2, y2]

    Returns:
        features[class_name] = list of instance features (torch tensor)
    '''
    features = {}

    if feat_extractor_name == 'DINOV2' : 
         extractor = get_dinov2_features 
    else:
         raise ValueError(f"Unsupported feature extractor: {feat_extractor_name}")
  

    for cls, samples in tqdm(support_data.items(), desc='Novel Memory Bank'):
    #for cls, samples in support_data.items():
    
        cls_feats = []

        for sample in samples:
            img_path = os.path.join(data_dir, sample['image'])
            pil_img = Image.open(img_path).convert('RGB')
            x, y, w, h = sample['bbox']
            #For coco format
            ref_boxes = [x, y, x+w, y+h]
            #For pascal format
            #ref_boxes = [x, y, w, h]
            
            # 1. Extract masks using SAM
            mask_np = extract_masks_from_support(sam2_predictor, pil_img, ref_boxes, device)

            # 2. Get DINOv2 features for full image
            full_feat = extractor(feat_extractor, image_transform, pil_img, device=device)  # (1, C, H_feat, W_feat)
            
            # 3. Resize mask to match feature map shape (H_feat, W_feat)
            resized_mask = resize_mask_to_features(mask_np, full_feat.shape[2:])  # only H, W
            resized_mask_tensor = torch.from_numpy(resized_mask).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

            masked_feat = full_feat * resized_mask_tensor
            valid_pixel_count = resized_mask_tensor.sum()

            if valid_pixel_count > 0:
                feat_vec = masked_feat.sum(dim=[2, 3]) / valid_pixel_count
                cls_feats.append(feat_vec.squeeze(0).detach().cpu())

        if len(cls_feats) > 0:
            proto = torch.stack(cls_feats, dim=0).mean(dim=0)
            features[cls] = [proto]  
        else:
            print(f"[Warning] No valid features for class {cls}")
            features[cls] = []


    return features

def compute_prototype_weights(memory_bank, device):
    """
    features_per_class: dict[class_name] = list of torch.Tensor features (each list contains only one proto)
    returns: dict[class_name] = prototype tensor, list[class_name] = class_names (for backward compatibility)
    """
    proto_cls_list = []
    proto_feat = []
    for cls, feats in memory_bank.items():
        if len(feats) > 0:  
            proto = feats[0]  
            proto_feat.append(proto.to(device))
            proto_cls_list.append(cls)
        else:
            print(f"[Warning] No features for class {cls}, skipping.")
    
    # Return both the normalized features and the list of class names
    # This maintains backward compatibility with existing code
    return F.normalize(torch.stack(proto_feat, dim=1), dim=0), proto_cls_list