import argparse
import os
import glob

from mmdet.apis import init_detector, inference_detector


def parser():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument(
        "--img_dir",
        dest="img_dir",
        help="Path to the original satellite image",
        type=str,
        required=True
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        help="Path to the output path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--model_cfg",
        dest="model_cfg",
        help="Path to the config file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint",
        help="Path to the checkpoint file",
        type=str,
        required=True
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parser()
    os.makedirs(args.output_dir, exist_ok=True)

    png_list = glob.glob(os.path.join(args.img_dir, "*.png"))

    model = init_detector(args.model_cfg, args.checkpoint, device='cuda:0')

    for png in png_list:
        out_file = os.path.join(args.output_dir, os.path.basename(png))
        result = inference_detector(model, png)
        model.show_result(png, result, out_file=out_file)