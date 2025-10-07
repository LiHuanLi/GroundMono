# GroundMono: Self-supervised Monocular Depth Estimation for Dynamic Objects with Ground Propagation

A framework for self-supervised monocular depth estimation, building on the Monodepth2/LiteMono architecture with enhanced ground propagation techniques. This work is accepted at IROS 2025.


## Overview
This repository contains code and dataset splits for training and evaluating monocular depth estimation models. The implementation is based on [Monodepth2](https://github.com/nianticlabs/monodepth2) and [LiteMono](https://github.com/noahzn/Lite-Mono) , and extends it with additional ground propagation module to improve depth estimation performance.


## Citation
If you use this work, please cite our IROS 2025 paper (to be updated with full citation after publication) :
```bibtex
@inproceedings{groundmono_2025,
  title={Self-supervised Monocular Depth Estimation for Dynamic Objects with Ground Propagation},
  author={Huan Li , Matteo Poggi, Fabio Tosi, Stefano Mattoccia},
  booktitle={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2025}
}

```

## Setup

### Prerequisites
- Anaconda or Miniconda environment
- Python 3.8.0 (recommended for compatibility)
- PyTorch 0.13.1+ with CUDA support


## Dataset Splits
The `splits/` directory contains predefined training and validation splits for different tasks


## Pretrained Models
Pretrained models can be downloaded from the following links:
- [GroundMono (monodepth2)](https://drive.google.com/file/d/1PqqDBA20ZRTocvqtt8qQAwTzhqy6yIKA/view?usp=sharing) 
- [GroundMono (litemono)](https://drive.google.com/file/d/1abZK5w6bx_GWaBH8z_e9MZD66aLq5mTc/view?usp=sharing)

## Object Mask Dataset for KITTI
The object mask dataset used in this project can be downloaded from: [coming soon]. This dataset contains semantic segmentation results for all the objects in KITTI sequences, which are extracted by the [Mask_RCNN](https://github.com/matterport/Mask_RCNN).


## Usage

### Training
To train the model on KITTI Odometry dataset:
```bash
python train.py --data_path /path/to/kitti_dataset --split eigen_zhou --base_model monodepth2
```

Key training parameters:
- `--data_path`: Path to KITTI dataset root
- `--split`: Dataset split (e.g., `eigen_zhou`, `benchmark`)
- `--model_name`: Name for saving training logs and weights
- `--base_model`: Base model architecture for ground propagation (e.g., litemono, monodepth2)


### Evaluation
To evaluate a trained model on the benchmark split:
```bash
python evaluate.py --data_path /path/to/kitti_dataset --load_weights_folder /path/to/trained_weights --eval_split eigen_benchmark  --base_model monodepth2
```

Key evaluation parameters:
- `--load_weights_folder`: Path to trained model weights
- `--eval_split`: Evaluation split (e.g., `benchmark`)
- `--base_model`: Base model architecture for ground propagation (e.g., litemono, monodepth2)


### Prediction for a Single Image
To generate depth prediction for a single image using a trained model:
```bash
python predict.py --image_path /path/to/input_image.jpg --load_weights_folder /path/to/trained_weights --output_path /path/to/save_prediction --base_model monodepth2

```

Parameters:
- `--image_path`: Path to input image
- `--load_weights_folder`: Path to trained model weights
- `--output_path`: Directory to save depth prediction
- `--pred_depth_format`: Output format (e.g., `png`, `npy`)
- `--base_model`:  Base model architecture for ground propagation (e.g., litemono, monodepth2)


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Note: The original Monodepth2 code is subject to Niantic's license terms. See the [license file](LICENSE) for full details.


## Acknowledgements
This work is based on [Monodepth2](https://github.com/nianticlabs/monodepth2) by Godard et al. We thank the authors for their open-source implementation.