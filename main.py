import argparse
import json
import torch
import torch.nn.functional
import model.dinov2
import model.sam2
import support_util
import query_util
import metric
from chatrex.upn import UPNWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot VOC evaluation with DINOv2 + SAM2")

    parser.add_argument('--json_path', type=str,
                        default='./data/PascalVOC/vocsplit/seed0/box_10shot_train.json',
                        help='Path to support set JSON file (default: %(default)s)')

    parser.add_argument(
                        '--feat_extractor_name',
                        type=str,
                        default='DINOV2',
                        choices=['DINOV2'],
                        help='feature extractor name (default: %(default)s)')

    parser.add_argument(
                        '--model_version',
                        type=str,
                        default='dinov2_vitl14',
                        choices=[
                            'dinov2_vits14', 'dinov2_vits14_reg',
                            'dinov2_vitb14', 'dinov2_vitb14_reg',
                            'dinov2_vitl14', 'dinov2_vitl14_reg',
                            'dinov2_vitg14', 'dinov2_vitg14_reg',
                        ],
                        help='model version (default: %(default)s)')

    parser.add_argument('--repo_or_dir', type=str,
                        default="./dinov2",
                        help='Repo or directory for dinov2 code (default: %(default)s)')

    parser.add_argument('--dinov2_checkpoint_dir', type=str,
                        default="./checkpoints",
                        help='Directory to pretrained dinov2 checkpoint (default: %(default)s)')

    parser.add_argument('--sam2_model_type', type=str,
                        default='large',
                        help='SAM2 model type (small/medium/large) (default: %(default)s)')

    parser.add_argument('--data_dir', type=str,
                        default='./data/',
                        help='Root directory for dataset (default: %(default)s)')

    parser.add_argument('--dinov2_image_size', type=int,
                        default=630,
                        help='Input size for dinov2 images (default: %(default)s)')

    parser.add_argument('--test_json', type=str,
                        default='./data/PascalVOC/VOC2007Test/voc_split1.json',
                        help='COCO format test json (default: %(default)s)')

    parser.add_argument('--test_img_dir', type=str,
                        default='./data/coco/val2017',
                        help='Directory containing test images (default: %(default)s)')

    parser.add_argument('--pred_json', type=str,
                        default='temp_pred.json',
                        help='Output prediction JSON file (default: %(default)s)')

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run models on (default: %(default)s)')
    
    parser.add_argument('--target_categories', type=str,nargs='+',
                        default=['bus','sofa','cow','bird','motorbike'],
                        help='Target categories for evaluation (default: %(default)s)')

    parser.add_argument('--min_threshold', type=float, default=0.01,
                        help='mean threshold for upn')

    parser.add_argument('--filter_by_categories', action='store_true',
                        help='filter by categories')

    parser.add_argument('--diffusion_steps', type=int, 
                        help='number of diffusion steps')

    parser.add_argument('--points_per_side', type=int,
                        default=32,
                        help='Points per side for SAM2 mask generator (default: %(default)s)')

    parser.add_argument('--alp', type=float, 
                        help='alpha in diffusion')
                        
    parser.add_argument('--lamb', type=float, 
                        help='lamda for decay')

    return parser.parse_args()

def main():
    args = parse_args()
    # Load UPN model if using UPN
    upn = None
    
    print('Loading UPN...')
    ckpt_path = './checkpoints/upn_large.pth'
    upn = UPNWrapper(ckpt_path)

    model_base_names = [
        'dinov2_vits14',
        'dinov2_vitb14', 
        'dinov2_vitl14',
        'dinov2_vitg14',
    ]

    model_name = args.model_version
    is_reg = model_name.endswith('_reg')

    # Remove '_reg' to get the base name
    base_name = model_name.replace('_reg', '')

    if base_name in model_base_names:
        suffix = 'reg4_pretrain.pth' if is_reg else 'pretrain.pth'
        checkpoint_filename = f"{base_name}_{suffix}"
        args.pretrained = f"{args.dinov2_checkpoint_dir}/{checkpoint_filename}"
    else:
        # For models not in the base names, construct a default path
        suffix = 'reg4_pretrain.pth' if is_reg else 'pretrain.pth'
        checkpoint_filename = f"{base_name}_{suffix}"
        args.pretrained = f"{args.dinov2_checkpoint_dir}/{checkpoint_filename}"
    
    if args.feat_extractor_name == 'DINOV2':
        print('Loading Dinov2...')
        feat_extractor, image_transform = model.dinov2.load_dinov2_model(
            args.device,
            args.model_version,
            image_size=(args.dinov2_image_size, args.dinov2_image_size),
            repo_or_dir=args.repo_or_dir,
            pretrained=args.pretrained
        )

    print('Loading SAM2...')
    sam2_model, sam2_predictor, sam2_mask_generator = model.sam2.load_sam2_components(
        model_type=args.sam2_model_type,
        device=args.device,
        points_per_side=args.points_per_side
    )
        
    # Load support set
    with open(args.json_path, 'r') as f:
        support_data = json.load(f)

    # Print stats
    for cls, instances in support_data.items():
        print(f"Class: {cls}, #Instances: {len(instances)}")

    # Build memory bank
    memory_bank = support_util.extract_support_features(
        support_data,
        sam2_predictor,
        args.feat_extractor_name,
        feat_extractor,
        image_transform,
        args.data_dir,
        args.device
    )

    proto_feat, proto_cls = support_util.compute_prototype_weights(memory_bank, args.device)
    min_th = 0.01
    
    # Load VOC2007 test loader
    image_paths, coco_style_loader = query_util.load_voc2007_coco_json(
        args.test_json,
        args.test_img_dir
    )

    # Generate predictions

    results = metric.generate_coco_style_predictions_upn(
        coco_style_loader,
        args.test_img_dir,
        sam2_predictor,
        args.feat_extractor_name,
        feat_extractor,
        image_transform,
        proto_feat,
        proto_cls,
        upn,  # Pass UPN model as parameter
        args.diffusion_steps,
        args.alp,
        args.lamb,
        args.device,
        args.min_threshold,
    )

    # Evaluate results
    metric.run_coco_eval(args.test_json, results, args.pred_json,target_categories=args.target_categories,filter_by_categories=args.filter_by_categories)


if __name__ == '__main__':
    main()

