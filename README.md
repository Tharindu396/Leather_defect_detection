# Leather Defect Detection

Mini-project for **EE7204/EC7205 — Image Processing and Computer Vision**, Department of Electrical and Information Engineering, University of Ruhuna.

The project detects and segments manufacturing defects (color stains, cuts, folds, glue marks, pokes) in leather images, with a focus on robustness under **low-light conditions**. It combines a diffusion-based synthetic data generator with a deep-learning segmentation model, and also includes a classical (non-ML) computer-vision baseline.

**Team:** Fonseka C.M. · Gunarathne A.N.G.T.N.B. · Piyumantha H.D.B.A.A. (Akila) · Rajapakshe R.M.M.L.D.

## Overview

Industrial leather inspection is often carried out under poor or inconsistent lighting, which degrades the accuracy of automated defect detectors. This project addresses that gap with the following pipeline:

1. **Simulate low-light conditions** on the [MVTec AD Leather](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset (gamma darkening, Gaussian noise, vignetting, uneven illumination).
2. **Generate synthetic low-light defective samples** with a DDPM (Denoising Diffusion Probabilistic Model) to balance the class distribution, since real defective samples are scarce.
3. **Segment defects** pixel-wise with an Attention U-Net (ResNet50 encoder, Squeeze-Excite blocks, deep supervision) trained on both normal and low-light images, with CLAHE contrast enhancement as a preprocessing step.
4. **Quantify defect severity** by converting predicted masks into damaged-area estimates via the area calculator.

A classical image-processing baseline (no deep learning) is also implemented for comparison, using reference-based anomaly detection, Gabor texture filtering, and multi-scale difference maps.

## Repository structure

| File | Description |
|---|---|
| [data_pipeline.ipynb](data_pipeline.ipynb) | Downloads/explores the MVTec AD Leather subset, applies low-light simulation, builds the preprocessing pipeline (resize, normalization, paired augmentation), creates train/val/test splits, and exposes a `tf.data.Dataset` factory. Verified with PSNR/SSIM. |
| [Synthetic_image_generation.ipynb](Synthetic_image_generation.ipynb) | Trains a DDPM (U-Net noise predictor, linear noise schedule, T=1000) on low-light defective images and samples new synthetic images to augment the defective class. Evaluated with PSNR/SSIM, saved to `synthetic_dataset/`. |
| [Leather_unet_multiclass_segmentation_tf.ipynb](Leather_unet_multiclass_segmentation_tf.ipynb) | **Main segmentation notebook.** Attention U-Net with a pretrained ResNet50 encoder, Squeeze-Excite blocks and deep supervision; weighted Focal + Dice loss with sqrt-inverse-frequency class weights; CLAHE preprocessing; full evaluation (per-class IoU/Dice, confusion matrix, failure-case analysis). |
| [Leather_unet_multiclass_segmentation_tf (2).ipynb](<Leather_unet_multiclass_segmentation_tf (2).ipynb>) | Alternate version of the segmentation notebook (low-light-robust UNet trained jointly on normal + gamma-degraded images, class-aware oversampling). |
| [Leather_detection_segmentation.ipynb](Leather_detection_segmentation.ipynb) | Notebook version of the classical CV segmentation pipeline (see `leather_defect_segmentation_cv.py`). |
| [leather_defect_segmentation_cv.py](leather_defect_segmentation_cv.py) | Pure image-processing baseline — **no ML model**. Reference-based anomaly detection, multi-scale Gaussian difference maps, Gabor filter bank, Laplacian-of-Gaussian, LAB color analysis, Otsu/percentile thresholding, morphological cleanup, per-defect-type tuned parameters. |
| [finetune_unet.py](finetune_unet.py) | Standalone (non-Colab) script to fine-tune the Attention U-Net locally — aggressive augmentation, focal+dice loss, class weighting, cosine-annealing LR with warm restarts. |
| [Calculator.ipynb](Calculator.ipynb) | Loads the trained model (`unet_results/best_attention_unet.keras`), predicts defect masks on new images, and computes the total damaged area per class. |
| `unet_results/best_attention_unet.keras` | Best checkpoint of the trained Attention U-Net segmentation model. |
| `unet_results_v3/clahe_comparison.png` | Visual comparison of images before/after CLAHE enhancement. |

## Dataset

Built on the **MVTec AD Leather** category:
- ~245 defect-free training images
- ~194 defective test images across 5 defect types: `color`, `cut`, `fold`, `glue`, `poke`
- Pixel-level ground-truth masks for defective images

Segmentation classes: `background(0)`, `color(1)`, `cut(2)`, `fold(3)`, `glue(4)`, `poke(5)`.

The raw dataset (`leather/`) and generated result folders (`segmentation_results/`, `segmentation_results_v2/`, `lowlight/`) are git-ignored — download the MVTec AD Leather category separately and place it as `leather/` at the repo root (`leather/train/good`, `leather/test/<defect_type>`, `leather/ground_truth/<defect_type>`).

Processed data produced by `data_pipeline.ipynb` is stored as float32 `.npy` arrays under `processed_dataset/{train,val,test}/{images,masks,lowlight}/` (images in `[0,1]`, masks as binary `{0,1}`).

## Methodology

- **Low-light simulation:** gamma darkening + Gaussian noise + vignetting + uneven illumination, applied to clean images to create paired normal/low-light training data.
- **CLAHE:** contrast-limited adaptive histogram equalization on the L channel (LAB space) to recover contrast lost to low-light degradation before feeding images to the segmentation model.
- **DDPM synthetic generation:** a U-Net-based noise predictor trained to reverse a forward diffusion process, used to synthesize additional low-light defective samples and balance the otherwise scarce defect classes.
- **Segmentation model:** Attention U-Net with a pretrained ResNet50 encoder, Squeeze-Excite blocks and deep supervision; loss = 0.6 × weighted Focal Cross-Entropy + 0.4 × weighted Dice, with square-root inverse-frequency class weights to counter background dominance.
- **Classical CV baseline:** no training required — anomaly maps are built from differences against a reference "good" image (multi-scale Gaussian, Gabor texture energy, LoG edges, LAB color deviation), fused per defect type, then thresholded and cleaned up morphologically.
- **Evaluation metrics:** IoU and Dice (segmentation quality, overall and per-class), PSNR and SSIM (synthetic/low-light image quality), confusion matrix, and failure-case analysis on worst-performing samples.

## Getting started

The notebooks are designed to run on **Google Colab** (GPU, e.g. T4) and expect the dataset to be available via Google Drive. Pipeline order:

1. `data_pipeline.ipynb`
2. `Synthetic_image_generation.ipynb`
3. `Leather_unet_multiclass_segmentation_tf.ipynb`
4. `Calculator.ipynb`

To run the classical CV baseline or fine-tune the model locally, use the standalone scripts:

```bash
python leather_defect_segmentation_cv.py
python finetune_unet.py
```

Update the `BASE_DIR` / `RESULTS_DIR` path constants at the top of each script to point to your local `leather/` dataset location before running.

### Dependencies

```
tensorflow
opencv-python
numpy
scikit-image
scikit-learn
matplotlib
pandas
```
