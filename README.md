# Memory-Augmented Autoencoder for Video Anomaly Detection

## Project Overview

This repository contains a modernized implementation of Memory-Augmented Autoencoder (MemAE) for video anomaly detection, based on the work by Gong et al. (2019). The implementation includes preprocessing enhancements, cross-dataset evaluation capabilities, and comprehensive performance analysis tools as part of dissertation research comparing temporal reasoning approaches in surveillance video analysis.

### Original Work Citation
```
Gong, D., Liu, L., Le, V., Saha, B., Mansour, M. R., Venkatesh, S., & Van Den Hengel, A. (2019). 
Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection. 
Proceedings of the IEEE/CVF International Conference on Computer Vision, 1705-1714.
```

**Original Repository:** https://github.com/donggong1/memae-anomaly-detection  
**License:** MIT (maintained from original work)

## Key Modifications & Enhancements

### Technical Upgrades:
- **MATLAB Preprocessing Integration**: Enhanced dataset preprocessing pipeline for optimal data format
- **Cross-Dataset Support**: Extended evaluation on PED1, PED2, and StreetScene datasets

## Repository Structure

```
dissertation_memory_augmentation_autoencoder/
├── data/                    # Preprocessed dataset directory
│   ├── ped1/               # Preprocessed PED1 training/testing data
│   ├── ped2/               # Preprocessed PED2 training/testing data
│   └── StreetScene/        # Preprocessed StreetScene data (optional)
├── datasets/               # Raw dataset processing and management scripts
├── matlab_script/          # MATLAB preprocessing utilities and functions
├── models/                 # Model architecture definitions and saved checkpoints
├── options/               # Training and testing configuration files
├── results/               # Generated results, metrics, and evaluation outputs
├── utils/                 # Utility functions and helper modules
├── visualization/         # Performance analysis and comparison scripts
│   ├── Performance_Comparison_PED1.py
│   ├── Performance_Comparison_PED2.py
│   └── README.md          # Visualization setup guide
├── script_training.py     # Main training script
├── script_testing.py      # Testing and evaluation script
├── script_eval_video.py   # Video-level evaluation utilities
└── README.md
```

## Dataset Setup & Preprocessing

### Required Datasets
- **UCSD PED1**: 36 training + 14 testing sequences
- **UCSD PED2**: 16 training + 12 testing sequences  
- **StreetScene**: 46 training + 35 testing sequences  (optional for advanced evaluation)

### MATLAB Preprocessing Requirements
The MemAE implementation requires MATLAB preprocessing for optimal data format and performance:

#### Preprocessing Steps:
```bash
# 1. Ensure MATLAB is installed and accessible
# 2. Place raw UCSD datasets in appropriate directories
# 3. Navigate to matlab_script/ directory
# 4. Run MATLAB preprocessing scripts in sequence:
#    - script_img_prep.m     # Process and prepare image data
#    - script_index_prep.m   # Generate sequence indices and metadata
# 5. Preprocessed data will be automatically saved in data/ directory
```

#### MATLAB Script Functionality:
- **Video Frame Extraction**: Converts video sequences to individual frames
- **Data Normalization**: Applies required preprocessing transformations
- **Format Standardization**: Ensures compatibility with PyTorch data loaders
- **Ground Truth Processing**: Prepares annotation data for evaluation

### Expected Directory Structure After Preprocessing
```
data/
├── ped1/
│   ├── Train/              # Training video frames
│   ├── Test/               # Testing video frames  
│   ├── Train_idx/          # Training sequence indices and metadata
│   ├── Test_idx/           # Testing sequence indices and metadata
│   └── Test_gt/            # Ground truth annotations for testing
├── ped2/
│   ├── Train/              # Training video frames
│   ├── Test/               # Testing video frames
│   ├── Train_idx/          # Training sequence indices and metadata  
│   ├── Test_idx/           # Testing sequence indices and metadata
│   └── Test_gt/            # Ground truth annotations for testing
└── StreetScene/
    ├── Train/              # Training video frames
    ├── Test/               # Testing video frames
    ├── Train_idx/          # Training sequence indices and metadata
    ├── Test_idx/           # Testing sequence indices and metadata
    └── Test_gt/            # Ground truth annotations for testing
```

#### Folder Contents:
- **Train/Test**: Preprocessed video frame data ready for model input
- **Train_idx/Test_idx**: Sequence index files and metadata for proper data loading
- **Test_gt**: Ground truth pixel-level annotations for anomaly evaluation and AUC calculation

### Dataset Download Sources
- **UCSD Datasets:** http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm
- **StreetScene:** Available through research collaboration

**Important Notes:**
- MATLAB preprocessing is required for proper data format compatibility
- Ensure sufficient disk space for preprocessed data (can be 3-5x original size)
- Remove system-generated hidden files before preprocessing

## Installation & Usage

### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/dineshgowri/dissertation_memory_augmentation_autoencoder.git
cd dissertation_memory_augmentation_autoencoder

# Create virtual environment
python3 -m venv memae_env
source memae_env/bin/activate  # On Windows: memae_env\Scripts\activate

# Install dependencies
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install tensorboard pillow numpy scipy scikit-learn
pip install Image
```

### 2. Dataset Preparation
```bash
# Ensure MATLAB is available for preprocessing
# Place raw datasets in appropriate directories
# Run MATLAB preprocessing scripts from matlab_script/ directory
# Verify preprocessed data is generated in data/ directory
```

### 3. Training
```bash
# Train on PED2 dataset (recommended for validation)
python ./script_training.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --Dataset UCSD_P2_256 \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --BatchSize 16 \
    --EpochNum 200 \
    --DataRoot /content/ \
    --ModelRoot ./results/ \
    --UseCUDA True

# Train on PED1 dataset
python ./script_training.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --Dataset UCSD_P1_256 \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --BatchSize 16 \
    --EpochNum 200 \
    --DataRoot /content/ \
    --ModelRoot ./results/ \
    --UseCUDA True

# Train on StreetScene (high computational requirements)
python ./script_training.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --Dataset StreetScene \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --BatchSize 16 \
    --EpochNum 200 \
    --DataRoot /content/ \
    --ModelRoot ./results/ \
    --UseCUDA True
```

### 4. Testing & Evaluation
```bash
# Test trained model on PED2
!python ./script_testing.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --Dataset UCSD_P2_256 \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --ModelRoot ./models/ \
    --DataRoot /content/ \
    --OutRoot ./results/ \
    --UseCUDA True

# Test trained model on PED1
!python ./script_testing.py \
    --ModelName MemAE \
    --ModelSetting Conv3DSpar \
    --Dataset UCSD_P1_256 \
    --MemDim 2000 \
    --EntropyLossWeight 0.0002 \
    --ShrinkThres 0.0025 \
    --ModelRoot ./models/ \
    --DataRoot /content/ \
    --OutRoot ./results/ \
    --UseCUDA True
```

## Citation & Academic Use

If you use this implementation in your research, please cite both the original paper and acknowledge this repository:

### Original Paper Citation:
```bibtex
@inproceedings{gong2019memorizing,
  title={Memorizing normality to detect anomaly: Memory-augmented deep autoencoder for unsupervised anomaly detection},
  author={Gong, Dong and Liu, Lingqiao and Le, Vuong and Saha, Budhaditya and Mansour, Moussa Reda and Venkatesh, Svetha and Van Den Hengel, Anton},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1705--1714},
  year={2019}
}
```

### Repository Acknowledgment:
```
This implementation builds upon the original MemAE codebase by Gong et al. (2019), 
with enhancements for cross-dataset evaluation, comprehensive performance analysis, 
and real-world deployment assessment as part of dissertation research on temporal 
reasoning approaches in video anomaly detection.

Original repository: https://github.com/donggong1/memae-anomaly-detection
Enhanced implementation: https://github.com/dineshgowri/dissertation_memory_augmentation_autoencoder
```

## Research Context

This work is part of Master's dissertation research.
