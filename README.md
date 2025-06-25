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

## Conclusion

Our project demonstrates that detecting FaceFusion-generated videos remains a challenging problem requiring multiple complementary approaches. Among our methods, the Temporal Synchronization approach achieved the best results on a small sample, with traditional feature extraction methods showing moderate effectiveness. The substantial overlap between classes across most feature spaces highlights the increasing sophistication of AI generation techniques.

### Performance Summary
- **HOG + SVM**: 67% accuracy (8/12 correct classifications)
- **Fourier + SVM**: 63% accuracy (76/120 correct classifications) 
- **Fourier + Logistic Regression**: 63.3% accuracy (76/120 correct classifications)
- **Laplacian + SVM**: 60.8% accuracy (73/120 correct classifications)
- **Laplacian + Logistic Regression**: 63.3% accuracy (76/120 correct classifications)
- **Temporal Synchronization**: 70% accuracy (7/10 correct classifications, smaller sample)
- **Vision Transformer**: 54% accuracy (27/50 correct classifications)

Each feature extraction method revealed different aspects of the detection challenge:

- **HOG features** captured gradient and edge information with moderate success (67% accuracy), but showed significant overlap between classes in the visualization space
- **Laplacian pyramid decomposition** provided multi-scale analysis that revealed some discriminative patterns (60-63% accuracy), but many fake videos still resembled real ones across scales
- **Temporal synchronization analysis** demonstrated potential (70% accuracy on a small sample), but showed that modern AI can increasingly maintain convincing temporal relationships
- **Fourier transform analysis** showed substantial overlap in frequency patterns between real and AI content (63% accuracy), indicating sophisticated frequency domain replication
- **Vision Transformer approach** underperformed expectations (54% accuracy), suggesting that even complex deep learning architectures struggle with the subtle artifacts in modern deepfakes

The consistent finding across all methods is that no single approach provides robust detection, as FaceFusion has become increasingly sophisticated at replicating authentic visual characteristics across multiple domains. The limited accuracy across all approaches (54-70%) demonstrates the fundamental challenge facing deepfake detection systems and highlights the need for continued research in this rapidly evolving field.

### Generalizability Considerations
A critical limitation of our study is the modest sample size and focus on a single deepfake generation method (FaceFusion). Future work should incorporate more diverse datasets spanning multiple generation algorithms, video qualities, and manipulation types to build more robust and generalizable detection systems.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.
