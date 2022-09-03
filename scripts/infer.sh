python src/tools/infer.py \
    --img_dir /home/ubuntu/workspace/ship_detection/dataset/inference/2022-08-27_sentinel1 \
    --output_dir output/yolof/inference \
    --model_cfg output/yolof_r50_c5_8x8_1x_coco.py \
    --checkpoint output/latest.pth \