# Show result images
python src/tools/test.py \
    --config output/yolof_r50_c5_8x8_1x_coco.py \
    --checkpoint output/latest.pth \
    --work-dir output/yolof \
    --show-dir output/yolof \

# # Output json files
# python src/tools/test.py \
#     --config output/yolof_r50_c5_8x8_1x_coco.py \
#     --checkpoint output/latest.pth \
#     --format-only \
#     --options "jsonfile_prefix=./output/yolof/" \
