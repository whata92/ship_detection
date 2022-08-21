# # Show result images
# python src/tools/test_mmdet.py \
#     --config output/yolof_r50_c5_8x8_1x_coco.py \
#     --checkpoint output/epoch_5.pth \
#     --work-dir output/yolof \
#     --show-dir output/yolof \

# Output json files
python src/tools/test_mmdet.py \
    --config output/yolof_r50_c5_8x8_1x_coco.py \
    --checkpoint output/epoch_5.pth \
    --format-only \
    --options "jsonfile_prefix=./output/yolof/" \
