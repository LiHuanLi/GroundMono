# GroundMono: Self-Supervised Monocular Depth Estimation with Ground Propagation

A framework for self-supervised monocular depth estimation, building on the Monodepth2 architecture with enhanced ground propagation techniques.

## Overview
This repository contains code and dataset splits for training and evaluating monocular depth estimation models, with a focus on improving performance through ground region propagation. The implementation is based on [Monodepth2](https://github.com/nianticlabs/monodepth2) and extends it with additional capabilities for handling ground plane information.

## Citation
If you use this work, please cite the original Monodepth2 paper:
```bibtex
@inproceedings{godard2019digging,
  title={Digging Into Self-Supervised Monocular Depth Estimation},
  author={Cl{\'{e}}ment Godard and
           Oisin {Mac Aodha} and
           Michael Firman and
           Gabriel J. Brostow},
  booktitle={The International Conference on Computer Vision (ICCV)},
  month={October},
  year={2019}
}
```

## Setup

### Prerequisites
- Anaconda or Miniconda environment
- Python 3.6.6 (recommended for compatibility)
- PyTorch 0.4.1+ with CUDA support

### Installation
```bash
# Create conda environment
conda create -n groundmono python=3.6.6 anaconda
conda activate groundmono

# Install dependencies
conda install pytorch=0.4.1 torchvision=0.2.1 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # Required for evaluation
```

For full dependency list, see [requirements.txt](requirements.txt).

## Dataset Splits
The `splits/` directory contains predefined training and validation splits for different tasks:

### Odometry Split (`splits/odom/`)
- `train_files.txt`: List of training samples for odometry tasks with format:
  ```
  [sequence_id] [frame_id] [camera_side]
  ```
  - `sequence_id`: Numeric identifier for the data sequence (0-8)
  - `frame_id`: Numeric identifier for the frame within the sequence
  - `camera_side`: `l` (left) or `r` (right) camera

### Benchmark Split (`splits/benchmark/`)
- Contains validation files with KITTI-style sequence formatting:
  ```
  [sequence_path] [frame_id] [camera_side]
  ```
  Example: `2011_09_26/2011_09_26_drive_0036_sync 640 r`

### Eigen Zhou Split (`splits/eigen_zhou/`)
- Standard split for monocular depth estimation evaluation, following Eigen et al. and Zhou et al. protocols.

## Usage

### Training
```bash
# Example training command
python train.py --data_path /path/to/dataset --split odom
```

### Evaluation
```bash
# Example evaluation command
python evaluate.py --data_path /path/to/dataset --load_weights_folder /path/to/weights --split benchmark
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

Note: The original Monodepth2 code is subject to Niantic's license terms. See the [license file](LICENSE) for full details.

## Acknowledgements
This work is based on [Monodepth2](https://github.com/nianticlabs/monodepth2) by Godard et al. We thank the authors for their open-source implementation.