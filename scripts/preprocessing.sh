for i in {1..9} ; do
    python3.9 src/tools/preprocessing.py \
    --img_path /workspace/dataset/LS-SSDD-v1.0-OPEN/JPEGImages/0${i}.jpg \
    --annotation_path /workspace/dataset/LS-SSDD-v1.0-OPEN/Annotations/0${i}.xml \
    --output_path dataset/cropped_512 \
    --cfg configs/training/default.yaml
done

for i in {10..15} ; do
    python3.9 src/tools/preprocessing.py \
    --img_path /workspace/dataset/LS-SSDD-v1.0-OPEN/JPEGImages/${i}.jpg \
    --annotation_path /workspace/dataset/LS-SSDD-v1.0-OPEN/Annotations/${i}.xml \
    --output_path dataset/cropped_512 \
    --cfg configs/training/default.yaml
done