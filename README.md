# SAR ship detection

## Overview

Ship detection for Sentinel-1 SAR image


## Installation

1. Install using `.settings/setup_poetry.sh`
```bash
bash .settings/setup_poetry.sh
source ~/.bash_profile
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

## Reference
