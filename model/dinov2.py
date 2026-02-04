import torch
from torchvision import transforms
import sys
import os

def get_dinov2_transform(image_size=(518, 518)):
    """Returns the preprocessing transform for DINOv2."""
    return transforms.Compose([
        transforms.Resize(size=image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def load_dinov2_model(device,
                      dinov2_model_name='dinov2_vitg14',
                      image_size=(518, 518),
                      repo_or_dir="./dinov2",
                      pretrained="./checkpoints/dinov2_vitg14_pretrain.pth"):
    """Loads the DINOv2 model and its preprocessing transform."""
    dinov2_dir = os.path.abspath(repo_or_dir)
    if dinov2_dir not in sys.path:
        sys.path.insert(0, dinov2_dir)

    # Normalize the pretrained flag/path handling:
    # - If pretrained is False -> don't load weights
    # - If pretrained is a string path and exists -> load from that checkpoint
    # - Otherwise try to rely on hub with pretrained=False (random init)

    print(f"Loading DINOv2 model: {dinov2_model_name}")
    print(f"Repo directory: {dinov2_dir}")
    print(f"Pretrained setting: {pretrained}")

    try:
        dinov2_model = torch.hub.load(
            repo_or_dir=dinov2_dir,
            model=dinov2_model_name,
            source='local',
            pretrained=False
        )

        # Optionally load checkpoint if a valid path is provided
        if isinstance(pretrained, str) and os.path.exists(pretrained):
            print(f"Loading pretrained weights from: {pretrained}")
            checkpoint = torch.load(pretrained, map_location='cpu')
            try:
                dinov2_model.load_state_dict(checkpoint, strict=False)
            except Exception as e:
                print(f"load_state_dict failed with strict=False: {e}")
                # Some repos nest weights under 'model' / 'state_dict'
                if isinstance(checkpoint, dict):
                    for k in ['model', 'state_dict', 'module']:
                        if k in checkpoint and isinstance(checkpoint[k], dict):
                            try:
                                dinov2_model.load_state_dict(checkpoint[k], strict=False)
                                print(f"Loaded weights from checkpoint['{k}'] with strict=False")
                                break
                            except Exception as e2:
                                print(f"Failed loading from checkpoint['{k}']: {e2}")
        else:
            if isinstance(pretrained, bool) and pretrained is True:
                print("Note: pretrained=True requested but no built-in weights available via local hub; using random init.")
            elif isinstance(pretrained, str):
                print(f"Warning: Pretrained weights not found at {pretrained}")
    except Exception as e:
        print(f"torch.hub.load failed: {e}")
        raise RuntimeError(f"Failed to load DINOv2 model {dinov2_model_name}: {e}")
    
    dinov2_model.eval()
    dinov2_model.to(device)

    dinov2_transform = get_dinov2_transform(image_size=image_size)

    return dinov2_model, dinov2_transform
