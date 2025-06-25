# Info290T-Deepfake-Detection

## Abhijith Varma Mudunuri and Atharva Patel

## Project Overview
This project aims to develop and compare methods for detecting AI-generated (deepfake) video frames. With the rise of deepfake technology, distinguishing between real and synthetic media is increasingly important. We implement multiple feature extraction techniques and classification models to identify patterns that differentiate authentic videos from those created using AI manipulation.

**Team Members:**
- Abhijith Varma Mudunuri
- Atharva Jayesh Patel
- Advisor: Hany Farid & Sarah Barrington

## Dataset
We use the Facefusion faceswap diffusion model within the Deepspeak v2 dataset, containing real and AI-generated (fake) face videos. Frames are extracted using InsightFace, resulting in a dataset of approximately 53,000 images, organized into `real` and `fake` categories.

### Sample Data
**Real Frames:**
![Real Frame 1](img/real/frame_000.jpg) ![Real Frame 2](img/real/frame_001.jpg) ![Real Frame 3](img/real/frame_002.jpg)

**Fake Frames:**
![Fake Frame 1](img/fake/frame_000.jpg) ![Fake Frame 2](img/fake/frame_001.jpg) ![Fake Frame 3](img/fake/frame_002.jpg)

## Repository Structure

- **Code/**: Python scripts for feature extraction, data processing, and modeling.
  - `analyzeHog.py`: HOG feature extraction and analysis.
  - `fourierExtract.py`: Fourier feature extraction and analysis.
  - `laplacianExtract.py`: Laplacian pyramid feature extraction and analysis.
  - `convert_to_mov.py`: Video conversion utility.
  - `data1.py`: Data handling utilities.
  - `VIT.py`: Vision Transformer (ViT) model implementation.

- **img/**: Example frames used in the project.
  - `real/`: Sample real frames.
  - `fake/`: Sample fake frames.

- **HOG/**: Results and visualizations for HOG-based detection.
  - `all_pca.png`, `all_tsne.png`: Dimensionality reduction visualizations.
  - `feature_importance.png`: Feature importance plots.
  - `confusion_matrix.png`: Model performance.
  - `results_summary.txt`: Summary of results.
  - `ALTERNATE RESULTS/`: Additional results and visualizations.

- **FOURIER/**: Results for Fourier-based detection.
  - Includes PCA/t-SNE plots, feature importance, confusion matrices, and summary.

- **LAPLACIAN/**: Results for Laplacian-based detection.
  - Includes PCA/t-SNE plots, feature importance, confusion matrices, and summary.

- **SYNCHRO/**: Results for temporal synchronization analysis.
  - Includes confusion matrices, feature importance, PCA/t-SNE, and ROC curve plots.
  - `SYNCRO_CLAUDABLE.py`: Synchronization analysis script.

- **FLICKER/**: Results for flicker inconsistency detection.
  - `FLICKER_CLAUDABLE.py`: Flicker analysis script.
  - `tsne_visualization.png`: Visualization of results.

- **VIT/**: Results for Vision Transformer-based detection.
  - `accuracy_plot.png`, `loss_plot.png`: Training curves.
  - `confusion_matrix.png`: Model performance.

- **index.ipynb**: Jupyter notebook with project overview, code snippets, and results.
- **index.html**: HTML export of the notebook for easy viewing.

## Feature Extraction & Detection Methods

### 1. Histogram of Oriented Gradients (HOG)
Captures gradient orientation distributions for shape analysis.

**Results:**
![HOG PCA](HOG/all_pca.png)
![HOG t-SNE](HOG/all_tsne.png)
![HOG Feature Importance](HOG/feature_importance.png)
![HOG Confusion Matrix](HOG/confusion_matrix.png)

### 2. Fourier Transform Analysis
Examines frequency domain characteristics for manipulation artifacts.

**Results:**
![Fourier PCA](FOURIER/all_pca.png)
![Fourier t-SNE](FOURIER/all_tsne.png)
![Fourier Feature Importance](FOURIER/feature_importance.png)
![Logistic Regression Confusion Matrix](FOURIER/logisticreg_confusion_matrix.png)
![SVM Confusion Matrix](FOURIER/svm_confusion_matrix.png)

### 3. Laplacian Pyramid Decomposition
Multi-scale analysis of image details.

**Results:**
![Laplacian PCA](LAPLACIAN/all_pca.png)
![Laplacian t-SNE](LAPLACIAN/all_tsne.png)
![Laplacian Feature Importance](LAPLACIAN/feature_importance.png)
![Laplacian Logistic Regression Confusion Matrix](LAPLACIAN/logreg_confusion_matrix.png)
![Laplacian SVM Confusion Matrix](LAPLACIAN/svm_confusion_matrix.png)

### 4. Temporal Synchronization Analysis
Consistency of motion between facial regions over time.

**Results:**
![Synchro PCA](SYNCHRO/pca_visualization.png)
![Synchro t-SNE](SYNCHRO/tsne_visualization.png)
![Synchro Feature Importance](SYNCHRO/feature_importance.png)
![Synchro Confusion Matrix](SYNCHRO/confusion_matrix.png)
![Synchro ROC Curve](SYNCHRO/roc_curve_comparison.png)

### 5. Flicker Inconsistency Detection
Frame-to-frame inconsistencies as manipulation indicators.

**Results:**
![Flicker t-SNE](FLICKER/tsne_visualization.png)

### 6. Vision Transformer (ViT)
Deep learning model for image classification.

**Results:**
![ViT Accuracy](VIT/accuracy_plot.png)
![ViT Loss](VIT/loss_plot.png)
![ViT Confusion Matrix](VIT/confusion_matrix.png)

## Results Summary
Each method's results are stored in their respective folders, including:
- Confusion matrices showing classification performance
- Feature importance plots highlighting key discriminative features
- Dimensionality reduction visualizations (PCA, t-SNE) showing data structure
- Detailed results summaries with quantitative metrics

## How to Use
1. Explore the `index.ipynb` notebook for a guided overview and code snippets.
2. Review the `Code/` folder for implementation details of each method.
3. Visualize results in the respective method folders.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
