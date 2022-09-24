# SAR ship detection

## Overview

Ship detection for Sentinel-1 SAR image.
In this repository, ship detection engine is provided.

The overall result can be accessed from web viewer as follows.
(Red rectangles are the detected ships)
![image](./.assets/ship_detection.gif)

Viewer's code will be available soon.


## Installation

1. Install using `.settings/setup_poetry.sh`
```bash
bash .settings/setup_poetry.sh
source ~/.bash_profile
```

2. Modify gdal version in `pyproject.toml`

Check installed `gdal` version

```bash
gdal-config --version
```

If you get version of `3.X.X`, modify gdal version in `pyproject.toml`.
If error occurs during its installation and the log says,
```
error in GDAL setup command: use_2to3 is invalid.
```

then, do following command.
```bash
poetry run pip install -U setuptools==57.5.0
```


3. Activate poetry bash
```bash
poetry shell
```

4. Install `torch`, `torchvision`, `mmcv-full` and `mmdet` manually
(Related issue: https://github.com/python-poetry/poetry/issues/2543)

It depends on the cuda version of your system.
```bash
pip install torch==1.10.0+cu102 torchvision==0.11.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
pip install mmcv-full
pip install mmdet
```

## Dataset
- [LS-SSDD-v1.0](https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN)


## Preprocessing
```bash
bash scripts/preprocessing.sh
```

## Training

```bash
python src/tools/train.py --cfg configs/training/default.yaml
```

- log file is stored in `/workspace/logs`
- checkpoints are created in the `/workspace/checkpoints`


## Integrating with Mapbox

1. Create [Mapbox](https://www.mapbox.com/) account
2. Create token for the project
    - User need to check `TILESETS.LIST`, `TILESETS.READ` and `TILESETS.WRITE`
3. Copy token for your environment variable.
```bash
export MAPBOX_ACCESS_TOKEN=<YOUR_MAPBOX_ACCESS_TOKEN>
```
4. Run following file first
```bash
bash scripts/initial_upload_to_mapbox.sh
```
5. After second trial, run following command
```bash
bash scripts/update_mapbox.sh
```

## Reference
- [Get started using Mapbox Tiling Service and the Tilesets CLI](https://docs.mapbox.com/help/tutorials/get-started-mts-and-tilesets-cli/)
