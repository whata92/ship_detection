from mmdet.apis import init_detector, inference_detector

target = "12_6953_16360_7465_16872.jpg"
# Specify the path to model config and checkpoint file
config_file = '/home/ubuntu/workspace/ship_detection/model_configs/yolof/yolof_r50_c5_8x8_1x_coco.py'
checkpoint_file = '/home/ubuntu/workspace/ship_detection/output/epoch_1.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = f'/home/ubuntu/workspace/ship_detection/dataset/cropped_512/{target}'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# or save the visualization results to image files
model.show_result(img, result, out_file=target)