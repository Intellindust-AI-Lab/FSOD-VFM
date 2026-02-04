CUDA_VISIBLE_DEVICES=0 python ./main.py \
--json_path ./data/CDFSOD/NEU-DET/annotations/1_shot_converted.json \
--test_json ./dataset/CDFSOD/NEU-DET/annotations/test.json \
--test_img_dir ./dataset/CDFSOD/NEU-DET/test/ \
--data_dir ./dataset \
--model_version  dinov2_vitl14 \
--feat_extractor_name DINOV2 \
--repo_or_dir dinov2 \
--dinov2_checkpoint_dir  ./checkpoints \
--min_threshold 0.01 \
--diffusion_steps 30 \
--alp 0.3 \
--lamb 0.5 

