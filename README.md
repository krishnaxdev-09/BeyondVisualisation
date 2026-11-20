# Beyond Visual Explanations: XAI Faithfulness for Pneumonia Detection

This project fine-tunes **ResNet50** and **DenseNet121** to detect pneumonia in pediatric chest X-rays. It then explains each prediction with **Grad-CAM**, **LIME** and **Integrated Gradients (IG)** and measures how faithful those explanations are using deletion and insertion AUC.

## Dataset

Kaggle: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (Kermany et al., 2018). It has 5,856 images in two classes, NORMAL and PNEUMONIA.

There are two ways to provide the data:

1. **Use a local copy.** Put the dataset in a folder called `chest_xray/` next to the notebook. The folder must contain `train/`, `val/` and `test/`.
2. **Download automatically.** If that folder is missing, the notebook downloads the dataset with `kagglehub`. This needs a Kaggle account and API token.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook Pneumonia_XAI_Final.ipynb
```

Run all cells from top to bottom. You need a CUDA GPU; with one, the full run takes about 1–2 hours.

If the run stops partway through the faithfulness cell, run that cell again. It continues from where it stopped.

To skip training and reuse saved models, set `RUN_TRAINING = False` in the configuration cell. The weights must already be in `results/models/`.

## Pipeline

| Step | Details |
|---|---|
| Data split | The `train` and `val` folders are merged, then split into a stratified 90/10 train/validation set (seed 42). The `test` set is left unchanged (624 images). |
| Class imbalance | WeightedRandomSampler |
| Augmentation | Resize to 224×224, horizontal flip, rotation of ±10°, brightness/contrast jitter |
| Stage 1 | Only the classification head is trained: 5 epochs, Adam, learning rate 1e-3 |
| Stage 2 | All layers are fine-tuned: up to 20 epochs, Adam, learning rate 1e-5, cosine annealing. Training stops early if validation loss does not improve for 5 epochs. |
| XAI | Grad-CAM (last convolutional block), LIME (1,000 samples), IG (300 steps) |
| Faithfulness | Deletion and insertion AUC over 50 steps, using a blurred image as the fill. A random-saliency control is included for comparison. |
| Statistics | Mean ± standard error, bootstrap 95% confidence intervals, paired Wilcoxon tests with Holm correction |
| IG baselines | Black, white, Gaussian noise and Gaussian blur. For each one the notebook records attribution statistics, sparsity, completeness error and deletion/insertion AUC. |

## Outputs (`results/`)

| File | Contents |
|---|---|
| `models/ResNet50.pth`, `models/DenseNet121.pth` | Trained weights |
| `classification_metrics.csv` | Accuracy, recall, precision, F1, AUC-ROC and confusion-matrix counts, at threshold 0.5 and at a threshold tuned on the validation set |
| `faithfulness_per_image.csv`, `faithfulness_summary.csv` | Deletion and insertion AUC for every model, method and image, plus summary statistics |
| `wilcoxon_tests.csv` | Paired significance tests |
| `ig_baselines_summary.csv` | Results of the IG baseline study |
| `all_results.json` | Every number reported in the paper, in one file |
| `figures/` | Confusion matrices, ROC curves, learning curves, faithfulness curves and bars, IG baseline plots, XAI examples |

The notebook also writes a `Data/` folder with 40 sample images, `sample_records.csv` and `data_dictionary.md`.

## Authors

Divyadeep Mishra, Krishna — School of Computer Science and Engineering
