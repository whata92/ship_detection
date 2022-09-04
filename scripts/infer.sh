python src/tools/infer.py \
    --img_dir /home/ubuntu/workspace/ship_detection/dataset/inference/2022-08-27_sentinel1 \
    --output_dir output/faster_rcnn/inference \
    --model_cfg output/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py \
    --checkpoint output/faster_rcnn/latest.pth \