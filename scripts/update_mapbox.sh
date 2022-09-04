# Create geojson.ld
python src/tools/convert_geojson_to_geojsonld.py \
    --geojson /home/ubuntu/workspace/ship_detection/output/faster_rcnn/inference/2022-08-27_sentinel1.geojson \
    --geojsonld /home/ubuntu/workspace/ship_detection/output/faster_rcnn/inference/2022-08-27_sentinel1.geojson.ld

# Upload to mapbox
tilesets upload-source wakuhatakeyama \
    sentinel1-ship \
    --replace \
    /home/ubuntu/workspace/ship_detection/output/faster_rcnn/inference/2022-08-27_sentinel1.geojson.ld

tilesets publish wakuhatakeyama.sentinel1-ship
