# SAR ship detection

## Overview

Ship detection for Sentinel-1 SAR image


## Installation

1. Modify `.docker/setup.sh`
2. Modify `.docker/requirement.txt`
3. Run setup commands
```bash
bash .docker setup.sh
```

## Dataset
- [LS-SSDD-v1.0](https://github.com/TianwenZhang0825/LS-SSDD-v1.0-OPEN)


## Training

```bash
python3.9 src/tools/train.py --cfg configs/training/default.yaml
```

- log file is stored in `/workspace/logs`
- checkpoints are created in the `/workspace/checkpoints`

## Reference
