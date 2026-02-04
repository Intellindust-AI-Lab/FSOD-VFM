import os
import json
import torch
import support_util
import query_util
from tqdm import tqdm
import PIL.Image
import numpy as np 
from pycocotools import mask as mask_utils
import pycocotools.coco
import pycocotools.cocoeval
import torch.nn.functional as F
import time
import numpy as np
from scipy.linalg import eigh
from collections import defaultdict
import copy
from torchvision.ops import batched_nms
import cv2



def get_category_id_to_name(coco_style_loader):
    """
    Given a COCO object, return a mapping from category_id to category_name.
    """
    cats = coco_style_loader.loadCats(coco_style_loader.getCatIds())
    return {cat['id']: cat['name'] for cat in cats}

def graph_diffusion_ios(masks_binary, labels, class_num, max_iter, alpha, rank_score=True,
                     tol=1e-6):
    
    n_masks = masks_binary.shape[0]
    masks = masks_binary.reshape(n_masks, -1).to(dtype=torch.float32)
    rw_ios = torch.zeros((n_masks,), device=masks_binary.device, dtype=torch.float32)
    
    if n_masks == 1:
        return rw_ios
    
    for cat_ind in range(class_num):
        cat_ind_tensor = torch.tensor(cat_ind, device=labels.device, dtype=labels.dtype)
        select_idxs = (labels == cat_ind_tensor)
        
        if select_idxs.sum() == 0:
            continue
            
        _masks = masks[select_idxs]
        if _masks.shape[0] == 0:
            continue
            
        n_cat = _masks.shape[0]
        
        # Compute base IoU matrix
        pos_num = _masks.sum(dim=-1).to(dtype=torch.float32)
        pos_num = torch.clamp(pos_num, min=1e-6)
        
        inter_num = _masks @ _masks.t()
        inter_num.fill_diagonal_(0.0)
        if rank_score:
            inter_num = torch.tril(inter_num, diagonal=0)
        
        # Normalized IoU matrix

        iou_matrix = inter_num / pos_num[:, None]

        personal_vector = iou_matrix.max(dim=-1)[0]   

        P = torch.zeros_like(iou_matrix)
        
        # Normalize each row to create probability distribution
        row_sums = iou_matrix.sum(dim=1)  # [n_cat] - remove keepdim=True
        valid_rows = row_sums > 1e-10     # [n_cat] - use small threshold to avoid zero division
        
        if valid_rows.any():
            # use safe index way to avoid dimension mismatch
            valid_indices = torch.where(valid_rows)[0]
            for idx in valid_indices:
                P[idx] = iou_matrix[idx] / valid_rows[idx]
        
        # Initialize stationary distribution
        pi = torch.ones(n_cat, device=P.device, dtype=P.dtype) / n_cat


        for iter_idx in range(max_iter):
            pi_old = pi.clone()
            
            # Random walk step
            pi = alpha * (P @ pi) + (1 - alpha) * personal_vector

            if torch.norm(pi - pi_old) < tol:
                break

        rw_ios[select_idxs] += pi
       
    return rw_ios

def generate_coco_style_predictions_upn(coco_style_loader,
                                    image_root_dir,
                                    sam2_mask_predictor,
                                    feat_extractor_name,
                                    feat_extractor,
                                    image_transform,
                                    proto_feat,
                                    proto_cls,
                                    upn,  # UPN model passed from main.py
                                    diffusion_steps,
                                    alp,
                                    lamb,
                                    device='cuda',
                                    min_threshold=0.01,
                                    ):  
    """
    Args:
        coco_style_loader: COCO object for VOC2007 test set.
        image_root_dir: root directory where image files are stored.
        sam2_mask_predictor: initialized SAM2 mask predictor.
        feat_extractor_name: name of feature extractor (DINOV2).
        feat_extractor: feature extractor, e.g. DINOv2 model.
        image_transform: preprocessing transform for feat_extractor.
        proto_feat: prototype feature (tensor).
        proto_cls: prototype cls name.
        upn: initialized UPN model for proposal generation.
        diffusion_steps: number of diffusion steps.
        alp: alpha in diffusion.
        lamb: lamda for decay.
        device: torch or CUDA device.
        min_threshold: minimum threshold for proposal filtering.
    Returns:
        List of prediction dicts in COCO format.
    """
    
    id_to_name = get_category_id_to_name(coco_style_loader)
    name_to_id = {v: k for k, v in id_to_name.items()}
    batch_size = 32  # Process N boxes at a time in SAM2
    results = []
    if feat_extractor_name == 'DINOV2' : 
         extractor = support_util.get_dinov2_features 
    
    # upn info
    candid_prompt = ["fine_grained_prompt", "coarse_grained_prompt"]
    # UPN model is now passed as parameter from main.py
    # Load all image metadata from COCO
    for img_dict in tqdm(coco_style_loader.dataset['images'], desc='Generating predictions'):
    #for img_dict in coco_style_loader.dataset['images']:
        img_id = img_dict['id']
        file_name = img_dict['file_name']
        img_path = os.path.join(image_root_dir, file_name)
        
        try:
            img_pil = PIL.Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[Warning] Failed to load image {img_path}: {e}")
            continue

        proposals = upn.inference(img_pil, candid_prompt[1])
        proposals_coarse = upn.filter(proposals, min_score=0.01, nms_value=1)
        
        # chek proposals
        if proposals is None or len(proposals.get('original_xyxy_boxes', [])) == 0:
            continue

        else:
            proposals = upn.filter(proposals, min_score=min_threshold, nms_value=1)
            
    
            if len(proposals.get('original_xyxy_boxes', [])) == 0:
                proposals = proposals_coarse
                boxes = proposals['original_xyxy_boxes'][0]
                scores = proposals['scores'][0]
            else:
                boxes = proposals['original_xyxy_boxes'][0]
                scores = proposals['scores'][0]


        # 1. Extract DINOv2 feature map
        feat_map = extractor(feat_extractor, image_transform, img_pil, device=device)
        
        # 2. with the upn info, to get the candidate mask in iter
        sam2_mask_predictor.set_image(img_pil)

        # Sort boxes and scores by scores in descending order and take top100
        if len(boxes) > 0 and len(scores) > 0:
            # Create list of (score, box) pairs
            box_score_pairs = list(zip(scores, boxes))
            # Sort by score in descending order
            box_score_pairs.sort(key=lambda x: x[0], reverse=True)
            # The original version uses top500, the revised version changes to top100, which is faster and better.
            top_100_pairs = box_score_pairs[:100]
            # Unzip back to scores and boxes
            scores, boxes = zip(*top_100_pairs)
            scores = list(scores)
            boxes = list(boxes)

        # collect all the results of the current image
        img_results = []
        
        if len(boxes) > 0:

            # Clip all boxes to valid image region
            iw, ih = img_pil.size
            clipped_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                # Clip coordinates to image boundaries
                x1 = max(0, min(x1, iw))
                y1 = max(0, min(y1, ih))
                x2 = max(0, min(x2, iw))
                y2 = max(0, min(y2, ih))
                # Ensure valid bbox (width and height > 0)
                if x2 > x1 and y2 > y1:
                    clipped_boxes.append([x1, y1, x2, y2])
              
            boxes = clipped_boxes

            for i in range(0, len(boxes), batch_size):
                # Get current batch of boxes
                batch_end = min(i + batch_size, len(boxes))
                batch_boxes = boxes[i:batch_end]
                batch_scores = scores[i:batch_end]
                
                # Convert batch boxes to numpy array format expected by SAM2
                batch_boxes_array = np.array(batch_boxes)
                
                # Predict masks for this batch
                masks, mask_scores, masks_256 = sam2_mask_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=batch_boxes_array,
                    multimask_output=False  # Get one mask per box
                )
                
                # Process each mask and corresponding box in the batch
                for j, (bbox, score, mask, mask_score) in enumerate(zip(batch_boxes, batch_scores, masks, mask_scores)):
                    
                    # Handle different mask formats
                    if isinstance(mask, (list, tuple)) and len(mask) > 0:
                        mask_to_use = mask[0]
                    else:
                        mask_to_use = mask
                    
                    masks_resize = support_util.resize_mask_to_features(mask_to_use, feat_map.shape[2:])
                    masks_resize = torch.from_numpy(masks_resize).cuda()
                    masked_feat = feat_map * masks_resize
                    valid_pixel_count = masks_resize.sum()
                    feat_vec = F.normalize(masked_feat.sum(dim=[2, 3]) / (valid_pixel_count + 1e-7), eps=1e-2)
                    sims = feat_vec @ proto_feat                    
                    top_score, top_cls = torch.max(sims, dim=1)

                    cat_id = name_to_id.get(proto_cls[top_cls[0].item()])
                    if cat_id is None:
                        continue
                    
                    # Handle mask encoding - mask_utils.encode returns a list
                    encoded_mask = mask_utils.encode(np.asfortranarray(mask_to_use.astype(np.uint8)))
                    
                    
                    # If encoded_mask is a list, take the first element
                    if isinstance(encoded_mask, (list, tuple)) and len(encoded_mask) > 0:
                        encoded_mask = encoded_mask[0]
                    
                    # Now encoded_mask should be a dict, handle counts
                    if isinstance(encoded_mask, dict) and 'counts' in encoded_mask:
                        if isinstance(encoded_mask['counts'], bytes):
                            encoded_mask['counts'] = encoded_mask['counts'].decode('utf-8')
                    else:
                        print(f"[WARNING] encoded_mask is not a dict or missing counts: {type(encoded_mask)}")
                        continue

                    masks_for_ios = masks_resize.clone()
                    if masks_for_ios.dim() == 2:  # [H, W]
                        masks_for_ios = masks_for_ios.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    elif masks_for_ios.dim() == 3:  # [1, H, W]
                        masks_for_ios = masks_for_ios.unsqueeze(0)  # [1, 1, H, W]
                    
                    img_results.append({
                        'image_id': img_id,
                        'feat':feat_vec.to(torch.float32).cpu().numpy(),
                        'masks_for_ios':masks_for_ios.to(torch.float32).cpu().numpy(),
                        'category_id': cat_id,
                        'segmentation': encoded_mask,
                        'bbox': [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
                        'score': float(top_score.item()),
                    })

        # Graph Diffusion
        if img_results:
            # collect all the masks and related information for ios calculation
            all_masks = []
            all_categories = []
            all_object_sims = []
            all_feats = []
            
            for item in img_results:
                all_masks.append(torch.from_numpy(item['masks_for_ios']).cuda())
                all_categories.append(item['category_id'])
                all_feats.append(torch.from_numpy(item['feat']).cuda())
            
            # stack all the masks and features
            if all_masks:
                stacked_masks = torch.stack(all_masks, dim=0)  # [n_masks, 1, H, W]
                # fix the categories indexing problem: map the actual category_id to a continuous index
                unique_categories = list(set(all_categories))
                category_to_idx = {cat: idx for idx, cat in enumerate(unique_categories)}
                stacked_categories = torch.tensor([category_to_idx[cat] for cat in all_categories], 
                                               device=stacked_masks.device, dtype=torch.long)
                       
                # compute the ios between all the masks
                try:
                    # Extract scores for sorting if softmerge_sort is enabled

                    ios_result = graph_diffusion_ios(stacked_masks, stacked_categories, 
                                                 len(unique_categories), max_iter=diffusion_steps, alpha=alp)               
                                    
                    ios = ios_result
                    # Apply score decay normally when no sorting
                    for i, item in enumerate(img_results):
                        if i < len(ios):
                            score_decay = 1 - ios[i]
                            if score_decay < 0:
                                score_decay = torch.tensor(0.0)

                            item['score'] = float(item['score'] * torch.pow(score_decay, lamb))
             
                except Exception as e:
                    print(f"[Warning] compute_semantic_ios failed: {e}, skipping IoU computation")
                    
        # top 100
        if img_results:
            img_results.sort(key=lambda x: x['score'], reverse=True)
            top_100_img_results = img_results[:100]
            results.extend(top_100_img_results)

    return results

    
def run_coco_eval(gt_json_path, prediction_results, pred_json='temp_predictions.json', 
                   target_categories=None, filter_by_categories=True, save_results=True):

    # remove key to save storage and convert numpy arrays to lists
    for result in prediction_results:
        if 'feat' in result:
            del result['feat']
        if 'segmentation' in result:
            del result['segmentation']
        if 'masks_for_ios' in result:
            del result['masks_for_ios']    
        # Convert any remaining numpy arrays to lists
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()

    # Save prediction results to file
    with open(pred_json, 'w') as f:
        json.dump(prediction_results, f)

    # Load ground truth
    coco_gt = pycocotools.coco.COCO(gt_json_path)

    # Add required fields if missing
    if 'info' not in coco_gt.dataset:
        coco_gt.dataset['info'] = {"description": "Auto-added info"}
    if 'licenses' not in coco_gt.dataset:
        coco_gt.dataset['licenses'] = []
        
    # Determine if segmentation evaluation is possible
    has_segmentation = any(
        isinstance(ann.get("segmentation"), (list, dict)) and ann.get("segmentation")
        for ann in coco_gt.dataset.get("annotations", [])
    )

    # Load predictions
    coco_dt = coco_gt.loadRes(prediction_results)
    
    # Choose evaluation types
    eval_types = ['bbox']
    if has_segmentation:
        eval_types.append('segm')
    
    # Store evaluation results
    eval_results = {}
    
    if filter_by_categories and target_categories:
        for iou_type in eval_types:
            print(f"\n====== COCO Evaluation (Target): {iou_type.upper()} ======")
            coco_eval = pycocotools.cocoeval.COCOeval(coco_gt, coco_dt, iouType=iou_type)
            target_cat_ids = coco_gt.getCatIds(catNms=target_categories)
            print(f"target_cat_ids: {target_cat_ids}")
            print(f"target_categories: {target_categories}")
            coco_eval.params.catIds = target_cat_ids

            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Store results for this evaluation type
            eval_results[iou_type] = {
                'target_categories': target_categories,
                'stats': coco_eval.stats.tolist(),
                'precision': coco_eval.eval['precision'].tolist() if 'precision' in coco_eval.eval else None,
                'recall': coco_eval.eval['recall'].tolist() if 'recall' in coco_eval.eval else None,
                'scores': coco_eval.eval['scores'].tolist() if 'scores' in coco_eval.eval else None
            }
 
    else:
        for iou_type in eval_types:
            print(f"\n====== COCO Evaluation: {iou_type.upper()} ======")
            coco_eval = pycocotools.cocoeval.COCOeval(coco_gt, coco_dt, iouType=iou_type)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Store results for this evaluation type
            eval_results[iou_type] = {
                'stats': coco_eval.stats.tolist(),
                'precision': coco_eval.eval['precision'].tolist() if 'precision' in coco_eval.eval else None,
                'recall': coco_eval.eval['recall'].tolist() if 'recall' in coco_eval.eval else None,
                'scores': coco_eval.eval['scores'].tolist() if 'scores' in coco_eval.eval else None
            }
    
    # Save evaluation results to JSON file
    if save_results:
        import os
        import datetime
        
        # Create results directory if it doesn't exist
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"coco_eval_results_{timestamp}.json"
        results_path = os.path.join(results_dir, results_filename)
        
        # Prepare results data
        results_data = {
            'timestamp': timestamp,
            'gt_json_path': gt_json_path,
            'pred_json': pred_json,
            'target_categories': target_categories,
            'filter_by_categories': filter_by_categories,
            'evaluation_results': eval_results,
            'stats_description': {
                'AP': 'Average Precision',
                'AP50': 'Average Precision at IoU=0.50',
                'AP75': 'Average Precision at IoU=0.75',
                'APs': 'Average Precision for small objects',
                'APm': 'Average Precision for medium objects',
                'APl': 'Average Precision for large objects',
                'AR': 'Average Recall',
                'AR50': 'Average Recall at IoU=0.50',
                'AR75': 'Average Recall at IoU=0.75',
                'ARs': 'Average Recall for small objects',
                'ARm': 'Average Recall for medium objects',
                'ARl': 'Average Recall for large objects'
            }
        }
        
        # Save to JSON file
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n====== Evaluation Results Saved ======")
        print(f"Results saved to: {results_path}")
        
        # Also save a summary file
        summary_filename = f"coco_eval_summary_{timestamp}.json"
        summary_path = os.path.join(results_dir, summary_filename)
        
        summary_data = {
            'timestamp': timestamp,
            'gt_json_path': gt_json_path,
            'pred_json': pred_json,
            'target_categories': target_categories,
            'summary_stats': {}
        }
        
        for eval_type, results in eval_results.items():
            stats = results['stats']
            summary_data['summary_stats'][eval_type] = {
                'AP': float(stats[0]),
                'AP50': float(stats[1]),
                'AP75': float(stats[2]),
                'APs': float(stats[3]),
                'APm': float(stats[4]),
                'APl': float(stats[5]),
                'AR': float(stats[6]),
                'AR50': float(stats[7]),
                'AR75': float(stats[8]),
                'ARs': float(stats[9]),
                'ARm': float(stats[10]),
                'ARl': float(stats[11])
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"Summary saved to: {summary_path}")
    
    return eval_results









    
