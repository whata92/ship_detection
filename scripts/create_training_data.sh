HOMEPATH=~/workspace/ship_detection

for i in {1..9} ; do
    python src/tools/create_training_data.py \
    --img_path $HOMEPATH/dataset/JPEGImages/0${i}.jpg \
    --annotation_path $HOMEPATH/dataset/Annotations/0${i}.xml \
    --output_path dataset/cropped_512 \
    --cfg configs/training/default.yaml
done

for i in {10..15} ; do
    python src/tools/create_training_data.py \
    --img_path $HOMEPATH/dataset/JPEGImages/${i}.jpg \
    --annotation_path $HOMEPATH/dataset/Annotations/${i}.xml \
    --output_path dataset/cropped_512 \
    --cfg configs/training/default.yaml
done