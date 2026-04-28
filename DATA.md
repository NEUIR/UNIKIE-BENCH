# 📊 Data Preparation Guide

This document provides detailed instructions for preparing and processing the UniKIE dataset.

<div align="center">

[← Back to README](./README.md)

</div>

## 📥 Step 1: Manually Download Datasets

Due to license restrictions, you need to manually download the following datasets:

### 1. 📄 CELL

- **Download URL**: https://rrc.cvc.uab.es/?ch=21&com=downloads
- **Action required**: Register an account and download Task1 image dataset
- **File to download**: `task1_test_imgs.zip`
- **Save location**: `datasets_process/dataset_source/CELL/task1_test_imgs.zip`

### 2. 🧾 SROIE

- **Download URL**: https://rrc.cvc.uab.es/?ch=13&com=downloads
- **Action required**: Register an account and download Task3 image dataset
- **File to download**: `SROIE_test_images_task_3.zip`
- **Save location**: `datasets_process/dataset_source/SROIE/SROIE_test_images_task_3.zip`

### 3. 📋 DeepForm

- **Download URL**: https://duebenchmark.com/data
- **File to download**: `DeepForm.tar.gz`
- **Action required**: After extraction, place the extracted `DeepForm` folder in the following location
- **Save location**: `datasets_process/dataset_source/DeepForm/DeepForm/` (containing all PDF files)

### 4. 📑 DocILE

- **Download URL**: https://github.com/rossumai/docile
- **Action required**: Obtain permission and download the dataset
- **File to download**: `annotated-trainval.zip` (extract to find PDF files)
- **Save location**: `datasets_process/dataset_source/docile/pdfs/` (containing all PDF files)

---

## 📁 Step 2: Verify Directory Structure

After completing manual downloads, verify that the `datasets_process/dataset_source/` directory structure is as follows:

```
datasets_process/dataset_source/
├── CELL/
│   └── task1_test_imgs.zip
├── DeepForm/
│   └── DeepForm/
│       └── [PDF files]
├── docile/
│   └── pdfs/
│       └── [PDF files]
└── SROIE/
    └── SROIE_test_images_task_3.zip
```

> **Note:** Other datasets will be automatically downloaded by scripts.
