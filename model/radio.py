"""
RADIO (C-RADIO v4-H) as feature extractor, same interface as DINOv2:
- load_radio_model() -> (model, image_transform)
- get_radio_features(model, image_transform, pil_img, device) -> (1, C, H_feat, W_feat) NCHW, C=1280
"""
import os
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor
from PIL import Image


def _resize_with_aspect_ratio(img_pil, target_long_side=630, patch_size=16):
    """Resize PIL image: long side = target_long_side, dimensions multiple of patch_size."""
    w, h = img_pil.size
    ratio = w / max(h, 1)
    if w >= h:
        new_w = target_long_side
        new_h = int(target_long_side / ratio)
    else:
        new_h = target_long_side
        new_w = int(target_long_side * ratio)
    new_w = max((new_w // patch_size), 1) * patch_size
    new_h = max((new_h // patch_size), 1) * patch_size
    return img_pil.resize((new_w, new_h), resample=Image.BICUBIC)


class RadioImageTransform:
    """Holds RADIO preprocessing config; same role as dinov2_transform for API consistency."""
    def __init__(self, target_long_side=630, patch_size=16):
        self.target_long_side = target_long_side
        self.patch_size = patch_size


def load_radio_model(
    device,
    model_version="c-radio_v4-h",
    cache_root=None,
    source="local",
):
    """
    Load RADIO model from local cache. Returns (model, image_transform).
    image_transform is a RadioImageTransform for get_radio_features (target_long_side, patch_size).
    """
    if cache_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cache_root = os.path.join(script_dir, "..", "model_cache")
    cache_root = os.path.abspath(cache_root)
    torch_hub_dir = os.path.join(cache_root, "torch_hub")
    os.makedirs(torch_hub_dir, exist_ok=True)

    torch.hub.set_dir(torch_hub_dir)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    hf_cache = os.path.join(cache_root, "huggingface")
    os.makedirs(hf_cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("TRANSFORMERS_CACHE", hf_cache)

    local_repo = os.path.join(torch_hub_dir, "NVlabs_RADIO_main")
    if not os.path.isdir(local_repo):
        raise FileNotFoundError(
            f"RADIO repo not found at {local_repo}. Please run from project root or download RADIO there."
        )

    model = torch.hub.load(
        local_repo,
        "radio_model",
        version=model_version,
        skip_validation=True,
        source=source,
    )
    model.eval()
    model.to(device)

    # Same default as DINOv2 path: 630 long side, patch 16 for RADIO
    image_transform = RadioImageTransform(target_long_side=630, patch_size=16)
    return model, image_transform


def get_radio_features(radio_model, image_transform, pil_img, device="cuda"):
    """
    Extract RADIO spatial features in NCHW, same contract as get_dinov2_features.
    Returns: (1, C, H_feat, W_feat) with C=1280, H_feat = H_in/16, W_feat = W_in/16.
    """
    target = getattr(image_transform, "target_long_side", 630)
    patch_size = getattr(image_transform, "patch_size", 16)
    pil_img = _resize_with_aspect_ratio(pil_img, target_long_side=target, patch_size=patch_size)

    x = pil_to_tensor(pil_img).to(dtype=torch.float32, device=device)
    x = x.div_(255.0).unsqueeze(0)  # (1, 3, H, W), value in [0, 1]

    nearest_res = radio_model.get_nearest_supported_resolution(*x.shape[-2:])
    x = F.interpolate(x, size=nearest_res, mode="bilinear", align_corners=False)

    with torch.inference_mode():
        with torch.autocast(device.split(":")[0] if ":" in device else "cuda", dtype=torch.bfloat16):
            _, spatial_features = radio_model(x, feature_fmt="NCHW")
    # spatial_features: (1, 1280, H/16, W/16); return float32 for downstream
    return spatial_features.float()
