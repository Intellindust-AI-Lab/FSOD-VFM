from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.sam2_image_predictor import SAM2ImagePredictor


def get_sam2_model_cfg_and_ckpt_path(model_type='large'):
    """Returns the config and checkpoint path for a given SAM2 model type."""
    config_map = {
        'tiny':      ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
        'small':     ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
        'base_plus': ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
        'large':     ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
    }
    return config_map.get(model_type)


def load_sam2_components(model_type='large', device='cuda', points_per_side=32):
    """Loads the SAM2 model, predictor, and mask generator."""
    model_cfg, ckpt_path = get_sam2_model_cfg_and_ckpt_path(model_type)
    
    model = build_sam2(model_cfg, ckpt_path, device=device, apply_postprocessing=True).to(device)
    predictor = SAM2ImagePredictor(model)

    mask_generator = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        points_per_batch=128,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        crop_n_layers=0,
        box_nms_thresh=0.7, 
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0.0,
        use_m2m=True,
    )

    

    return model, predictor, mask_generator