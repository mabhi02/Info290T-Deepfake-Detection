# Info290T-Deepfake-Detection

## Abhijith Varma Mudunuri and Atharva Patel

## Project Overview
This project aims to develop and compare methods for detecting AI-generated (deepfake) video frames. With the rise of deepfake technology, distinguishing between real and synthetic media is increasingly important. We implement multiple feature extraction techniques and classification models to identify patterns that differentiate authentic videos from those created using AI manipulation.

**Team Members:**
- Abhijith Varma Mudunuri
- Atharva Jayesh Patel

## Dataset
We use the Facefusion faceswap diffusion model within the Deepspeak v2 dataset, containing real and AI-generated (fake) face videos. Frames are extracted using InsightFace, resulting in a dataset of approximately 53,000 images, organized into `real` and `fake` categories.

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
- **Histogram of Oriented Gradients (HOG)**: Captures gradient orientation distributions for shape analysis.
- **Fourier Transform Analysis**: Examines frequency domain characteristics for manipulation artifacts.
- **Laplacian Pyramid Decomposition**: Multi-scale analysis of image details.
- **Temporal Synchronization Analysis**: Consistency of motion between facial regions over time.
- **Flicker Inconsistency Detection**: Frame-to-frame inconsistencies as manipulation indicators.
- **Vision Transformer (ViT)**: Deep learning model for image classification.

## Results
Each method's results are stored in their respective folders, including:
- Confusion matrices
- Feature importance plots
- Dimensionality reduction visualizations (PCA, t-SNE)
- Results summaries

## How to Use
1. Explore the `index.ipynb` notebook for a guided overview and code snippets.
2. Review the `Code/` folder for implementation details of each method.
3. Visualize results in the respective method folders.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.