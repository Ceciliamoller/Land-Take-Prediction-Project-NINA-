# Land-Take-Prediction-Project-NINA-

## Project Overview
Predicting land-take (urban expansion, deforestation, etc.) from very high resolution (VHR) satellite imagery using deep learning. The project uses paired before/after RGBY satellite images along with binary land-take masks from the HABLOSS dataset.

## Current Progress

### Simple CNN Proof of Concept
A basic CNN model for binary semantic segmentation has been implemented in `notebooks/01_explore_habloss.ipynb`. The POC includes:
- **Preprocessing pipeline**: Patches sample images into 128×128 tiles with mask upsampling to match image resolution
- **Model architecture**: Simple encoder-decoder CNN with 3 encoding blocks, 2 decoding blocks, and ~500k parameters
- **Training setup**: 80/20 train/validation split with CrossEntropyLoss and Adam optimizer
- **Input**: 6-channel images (RGB before + RGB after)
- **Output**: 2-class segmentation (land-take vs. no change)

The notebook demonstrates the full pipeline from data loading through model training and evaluation visualization.

## Initial repo setup
- put messy one time stuff in notebooks folder
- put reusable code in src 
- config.py should contain path to data 

### data
- raw data from NINA should be in data/raw, and intermediate and processed in data/interm and data/processed 
- data/raw/vhr: actual satelite images, multi-band, before and after
- data/raw/masks: binary land take masks (names matching satelite images in vhr)

### src/data
- **HablossSampleDataset**: PyTorch Dataset class for loading VHR imagery and masks. Takes a list of file IDs and returns paired tensors of shape (C, H, W) for images and (H, W) for masks.