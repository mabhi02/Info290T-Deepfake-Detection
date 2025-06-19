import os
import numpy as np
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Define paths
BASE_PATH = r"C:\Users\athar\Documents\GitHub\testcv\faceData"
TRAIN_FAKE_PATH = os.path.join(BASE_PATH, "train", "fake")
TRAIN_REAL_PATH = os.path.join(BASE_PATH, "train", "real")
TEST_FAKE_PATH = os.path.join(BASE_PATH, "test", "fake")
TEST_REAL_PATH = os.path.join(BASE_PATH, "test", "real")

# Create directories for visualizations and results
RESULTS_DIR = "laplacian_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Modified limits as requested
TRAIN_FAKE_LIMIT = 250  # Limit to 250 fake videos for training
TRAIN_REAL_LIMIT = 250  # Limit to 250 real videos for training
TEST_FAKE_LIMIT = 60    # Limit to 60 fake videos for testing
TEST_REAL_LIMIT = 60    # Limit to 60 real videos for testing

def get_video_dirs(directory, limit=None, seed=RANDOM_SEED, exclude_dirs=None):
    """Get a specified number of video directories from a parent directory
    
    Args:
        directory: Directory containing video subdirectories
        limit: Maximum number of directories to return
        seed: Random seed for consistent selection
        exclude_dirs: List of directory paths to exclude
    """
    video_dirs = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            # Skip if this directory should be excluded
            if exclude_dirs and item_path in exclude_dirs:
                continue
            video_dirs.append(item_path)
    
    # Sort for deterministic order before random selection
    video_dirs.sort()
    
    if seed is not None:
        # Create a separate random generator instance
        rng = random.Random(seed)
        rng.shuffle(video_dirs)
    
    if limit and len(video_dirs) > limit:
        return video_dirs[:limit]
    
    return video_dirs

def build_laplacian_pyramid(image, levels=4):
    """Build a Laplacian pyramid for an image
    
    Args:
        image: Input image
        levels: Number of pyramid levels to generate
        
    Returns:
        A list containing the Laplacian pyramid levels
    """
    # Ensure image is grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to ensure consistent dimensions
    gray = cv2.resize(gray, (128, 128))
    
    # Convert to float32 for calculations
    gray = gray.astype(np.float32) / 255.0
    
    # Initialize pyramid
    pyramid = []
    current = gray.copy()
    
    # Build Gaussian pyramid
    gaussian_pyramid = [current]
    for i in range(levels - 1):
        current = cv2.pyrDown(current)
        gaussian_pyramid.append(current)
    
    # Build Laplacian pyramid
    for i in range(levels - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = gaussian_pyramid[i] - expanded
        pyramid.append(laplacian)
    
    # Add the smallest level of the Gaussian pyramid
    pyramid.append(gaussian_pyramid[-1])
    
    return pyramid

def extract_pyramid_features(pyramid):
    """Extract statistical features from each level of the Laplacian pyramid
    
    Args:
        pyramid: List of Laplacian pyramid levels
        
    Returns:
        Array of features
    """
    features = []
    
    # For each pyramid level, compute statistical features
    for level_idx, level in enumerate(pyramid):
        # Basic statistics
        mean = np.mean(level)
        std = np.std(level)
        median = np.median(level)
        min_val = np.min(level)
        max_val = np.max(level)
        
        # Energy features
        energy = np.sum(level**2)
        abs_energy = np.sum(np.abs(level))
        
        # Histogram features (10 bins)
        hist, _ = np.histogram(level, bins=10, range=(-1, 1))
        hist = hist / hist.sum() if hist.sum() > 0 else hist  # Normalize
        
        # Structural features
        kurtosis = np.mean((level - mean)**4) / (std**4) if std > 0 else 0
        skewness = np.mean((level - mean)**3) / (std**3) if std > 0 else 0
        
        # Spectral features
        fft = np.abs(np.fft.fft2(level))
        fft_energy = np.sum(fft**2)
        
        # Add level index as a feature multiplier (to differentiate between levels)
        level_factor = level_idx + 1
        
        # Add all features with level information
        level_features = [
            mean, std, median, min_val, max_val, 
            energy, abs_energy, kurtosis, skewness, fft_energy
        ]
        # Add histogram bins
        level_features.extend(hist)
        
        # Add level identifier prefix to the feature names (for reference)
        features.extend(level_features)
    
    return np.array(features)

def compute_laplacian_feature_names(levels=4):
    """Generate names for the Laplacian pyramid features"""
    feature_names = []
    
    # Create names for the statistical features
    stat_names = ['mean', 'std', 'median', 'min', 'max', 'energy', 'abs_energy', 
                 'kurtosis', 'skewness', 'fft_energy']
    
    for level in range(levels):
        for name in stat_names:
            feature_names.append(f'level{level+1}_{name}')
        
        # Add histogram bin names
        for bin_idx in range(10):
            feature_names.append(f'level{level+1}_hist_bin{bin_idx+1}')
    
    return feature_names

def extract_laplacian_temporal_features(video_dir, pyramid_levels=4):
    """Extract Laplacian pyramid features from all frames in a video directory and compute temporal features"""
    frame_files = [f for f in os.listdir(video_dir) if f.startswith('frame_') and f.endswith(('.jpg', '.png'))]
    frame_files.sort()  # Ensure frames are in order
    
    # Skip if too few frames
    if len(frame_files) < 3:
        print(f"Warning: Too few frames in {video_dir} - skipping")
        return None
    
    # Load and process all frames
    pyramid_features_list = []
    
    for frame_file in frame_files:
        try:
            frame_path = os.path.join(video_dir, frame_file)
            image = cv2.imread(frame_path)
            
            if image is None:
                print(f"Warning: Could not read {frame_path} - skipping")
                continue
                
            # Build Laplacian pyramid and extract features
            pyramid = build_laplacian_pyramid(image, levels=pyramid_levels)
            pyramid_features = extract_pyramid_features(pyramid)
            pyramid_features_list.append(pyramid_features)
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")
            continue
    
    # Convert to numpy array for easier computation
    pyramid_features_array = np.array(pyramid_features_list)
    
    # Skip if too few frames were successfully processed
    if len(pyramid_features_array) < 3:
        print(f"Warning: Too few valid frames in {video_dir} - skipping")
        return None
    
    # Calculate frame-to-frame differences (temporal information)
    pyramid_feature_diffs = np.abs(pyramid_features_array[1:] - pyramid_features_array[:-1])
    
    # Initialize feature dictionary
    features = {}
    
    # 1. Basic Statistical Features
    features['pyramid_mean'] = np.mean(pyramid_features_array)
    features['pyramid_std'] = np.std(pyramid_features_array)
    features['pyramid_median'] = np.median(pyramid_features_array)
    features['pyramid_min'] = np.min(pyramid_features_array)
    features['pyramid_max'] = np.max(pyramid_features_array)
    
    # 2. Temporal Variation Metrics
    features['pyramid_temporal_std'] = np.std(pyramid_features_array, axis=0).mean()  # Average std across time
    features['pyramid_temporal_mean_diff'] = np.mean(pyramid_feature_diffs)  # Average difference between frames
    features['pyramid_temporal_max_diff'] = np.max(pyramid_feature_diffs)  # Maximum difference
    features['pyramid_temporal_diff_std'] = np.std(pyramid_feature_diffs)  # Std of differences
    
    # 3. Consistency Metrics
    # Higher values may indicate real videos (more stable)
    features['pyramid_stability'] = 1.0 / (features['pyramid_temporal_std'] + 1e-6)
    
    # Measure of consistency across pyramid features (spatial uniformity)
    features['pyramid_uniformity'] = 1.0 / (np.std(np.mean(pyramid_features_array, axis=0)) + 1e-6)
    
    # Measure of consistency between adjacent frames
    consistency_scores = []
    for i in range(len(pyramid_features_array) - 1):
        corr = pearsonr(pyramid_features_array[i], pyramid_features_array[i+1])[0]
        if not np.isnan(corr):
            consistency_scores.append(corr)
    
    if consistency_scores:
        features['pyramid_temporal_consistency'] = np.mean(consistency_scores)
        features['pyramid_temporal_consistency_std'] = np.std(consistency_scores)
    else:
        features['pyramid_temporal_consistency'] = 0
        features['pyramid_temporal_consistency_std'] = 0
    
    # 4. Spectral Analysis (if enough frames)
    if len(pyramid_features_array) >= 8:  # Need minimum number of frames for FFT
        # Compute FFT for each pyramid dimension and average
        fft_magnitudes = []
        for i in range(pyramid_features_array.shape[1]):
            feature_series = pyramid_features_array[:, i]
            # Remove mean (DC component)
            feature_series = feature_series - np.mean(feature_series)
            
            # Compute FFT
            fft = np.abs(np.fft.rfft(feature_series))
            fft_magnitudes.append(fft)
        
        # Average FFT across all pyramid dimensions
        avg_fft = np.mean(fft_magnitudes, axis=0)
        
        # Calculate energy in different frequency bands
        n_bands = 3  # Low, mid, high
        band_size = len(avg_fft) // n_bands
        
        total_energy = np.sum(avg_fft)
        if total_energy > 0:  # Avoid division by zero
            features['pyramid_low_freq_energy'] = np.sum(avg_fft[:band_size]) / total_energy
            features['pyramid_mid_freq_energy'] = np.sum(avg_fft[band_size:2*band_size]) / total_energy
            features['pyramid_high_freq_energy'] = np.sum(avg_fft[2*band_size:]) / total_energy
            
            # Peak frequency information
            peak_freq_idx = np.argmax(avg_fft)
            features['pyramid_peak_freq_idx'] = peak_freq_idx
            features['pyramid_peak_freq_magnitude'] = avg_fft[peak_freq_idx] / total_energy
        else:
            features['pyramid_low_freq_energy'] = 0
            features['pyramid_mid_freq_energy'] = 0
            features['pyramid_high_freq_energy'] = 0
            features['pyramid_peak_freq_idx'] = 0
            features['pyramid_peak_freq_magnitude'] = 0
    else:
        # Not enough frames for spectral analysis
        features['pyramid_low_freq_energy'] = 0
        features['pyramid_mid_freq_energy'] = 0
        features['pyramid_high_freq_energy'] = 0
        features['pyramid_peak_freq_idx'] = 0
        features['pyramid_peak_freq_magnitude'] = 0
    
    # 5. Eigenvalue Analysis (if enough samples)
    if pyramid_features_array.shape[0] > 3 and pyramid_features_array.shape[1] > 3:
        try:
            # Center the data
            centered_features = pyramid_features_array - np.mean(pyramid_features_array, axis=0)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(centered_features, rowvar=False)
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
            
            # Only use positive eigenvalues
            eigenvalues = eigenvalues[eigenvalues > 0]
            
            if len(eigenvalues) > 0:
                total_variance = np.sum(eigenvalues)
                
                # Ratio of largest eigenvalue to sum (principal mode)
                features['pyramid_top_eigenvalue_ratio'] = eigenvalues[0] / total_variance if total_variance > 0 else 0
                
                # Ratio of top 3 eigenvalues (or fewer if not enough)
                top_n = min(3, len(eigenvalues))
                features['pyramid_top3_eigenvalue_ratio'] = np.sum(eigenvalues[:top_n]) / total_variance if total_variance > 0 else 0
                
                # Ratio between first and second eigenvalues (if available)
                if len(eigenvalues) > 1:
                    features['pyramid_eigenvalue_decay'] = eigenvalues[0] / eigenvalues[1]
                else:
                    features['pyramid_eigenvalue_decay'] = 1.0
            else:
                features['pyramid_top_eigenvalue_ratio'] = 0
                features['pyramid_top3_eigenvalue_ratio'] = 0
                features['pyramid_eigenvalue_decay'] = 1.0
        except (np.linalg.LinAlgError, ValueError) as e:
            # Handle potential numerical issues
            print(f"Eigenvalue calculation error in {video_dir}: {e}")
            features['pyramid_top_eigenvalue_ratio'] = 0
            features['pyramid_top3_eigenvalue_ratio'] = 0
            features['pyramid_eigenvalue_decay'] = 1.0
    else:
        features['pyramid_top_eigenvalue_ratio'] = 0
        features['pyramid_top3_eigenvalue_ratio'] = 0
        features['pyramid_eigenvalue_decay'] = 1.0
    
    # Add level-specific statistics
    # Calculate average features for each level of the pyramid
    feature_dim = pyramid_features_array.shape[1]
    features_per_level = feature_dim // pyramid_levels
    
    if features_per_level > 0:
        for level in range(pyramid_levels):
            start_idx = level * features_per_level
            end_idx = (level + 1) * features_per_level
            
            if end_idx <= feature_dim:  # Ensure indices are within bounds
                level_features = pyramid_features_array[:, start_idx:end_idx]
                
                # Add average statistics for this level
                features[f'level{level+1}_avg_mean'] = np.mean(level_features)
                features[f'level{level+1}_avg_std'] = np.std(level_features)
                features[f'level{level+1}_avg_min'] = np.min(level_features)
                features[f'level{level+1}_avg_max'] = np.max(level_features)
                
                # Add temporal consistency for this level
                level_diffs = np.abs(level_features[1:] - level_features[:-1])
                features[f'level{level+1}_temporal_change'] = np.mean(level_diffs)
                
                # Add noise estimates for this level (higher frequency levels show more noise in fakes)
                features[f'level{level+1}_noise_ratio'] = np.std(level_features) / (np.mean(np.abs(level_features)) + 1e-6)
    
    # 6. Laplacian Pyramid specific features (relationships between levels)
    if pyramid_levels > 1:
        for level in range(pyramid_levels - 1):
            level1 = level
            level2 = level + 1
            
            # Compare statistics between adjacent levels
            if features_per_level > 0:
                level1_start = level1 * features_per_level
                level1_end = (level1 + 1) * features_per_level
                level2_start = level2 * features_per_level
                level2_end = (level2 + 1) * features_per_level
                
                if level1_end <= feature_dim and level2_end <= feature_dim:
                    level1_features = pyramid_features_array[:, level1_start:level1_end]
                    level2_features = pyramid_features_array[:, level2_start:level2_end]
                    
                    # Calculate ratio of standard deviations between levels
                    # (This can detect inconsistencies in noise distribution across scales)
                    level1_std = np.std(level1_features)
                    level2_std = np.std(level2_features)
                    
                    if level2_std > 0:
                        features[f'level{level1+1}_to_{level2+1}_std_ratio'] = level1_std / level2_std
                    else:
                        features[f'level{level1+1}_to_{level2+1}_std_ratio'] = 0.0
                    
                    # Calculate correlation between adjacent levels
                    # (Measures how well the multi-scale structure is preserved)
                    level1_mean = np.mean(level1_features, axis=1)
                    level2_mean = np.mean(level2_features, axis=1)
                    
                    if len(level1_mean) > 1:  # Ensure we have enough samples
                        corr = pearsonr(level1_mean, level2_mean)[0]
                        if not np.isnan(corr):
                            features[f'level{level1+1}_to_{level2+1}_correlation'] = corr
                        else:
                            features[f'level{level1+1}_to_{level2+1}_correlation'] = 0.0
                    else:
                        features[f'level{level1+1}_to_{level2+1}_correlation'] = 0.0
    
    # Return all features as numpy array
    feature_names = list(features.keys())
    return np.array(list(features.values())), feature_names

def process_videos(video_dirs, label):
    """Process a list of video directories and extract features"""
    features_list = []
    labels = []
    video_names = []
    feature_names = None
    
    for video_dir in tqdm(video_dirs, desc=f"Processing {label} videos"):
        try:
            video_name = os.path.basename(video_dir)
            
            # Extract Laplacian pyramid features and temporal analysis
            result = extract_laplacian_temporal_features(video_dir)
            
            if result is not None:
                features, feature_names = result
                features_list.append(features)
                labels.append(1 if label == 'fake' else 0)  # 1 for fake, 0 for real
                video_names.append(video_name)
        except Exception as e:
            print(f"Error processing {video_dir}: {e}")
    
    if not features_list:
        return np.array([]), np.array([]), [], []
    
    return np.array(features_list), np.array(labels), video_names, feature_names

def visualize_tsne(features, labels, video_names, save_path):
    """Apply t-SNE and visualize the results"""
    # Handle case with small number of samples
    n_samples = features.shape[0]
    
    if n_samples <= 5:
        print(f"Warning: Not enough samples ({n_samples}) for t-SNE visualization. Skipping.")
        return
    
    # Set perplexity to min(30, n_samples/3) with a minimum of 2
    perplexity = min(30, max(2, n_samples // 3))
    print(f"Using perplexity of {perplexity} for {n_samples} samples in t-SNE")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, n_iter=2000, random_state=RANDOM_SEED)
    features_embedded = tsne.fit_transform(features)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'x': features_embedded[:, 0],
        'y': features_embedded[:, 1],
        'label': ['AI-Generated' if l == 1 else 'Real' for l in labels],
        'video': video_names
    })
    
    # Plot without text labels
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=100)
    
    plt.title('t-SNE Visualization of Laplacian Pyramid Temporal Features')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_pca(features, labels, video_names, save_path, feature_names):
    """Apply PCA and visualize the results"""
    # Handle case with small number of samples
    n_samples = features.shape[0]
    n_features = features.shape[1]
    
    if n_samples < 2:
        print(f"Warning: Not enough samples ({n_samples}) for PCA visualization. Skipping.")
        return None
    
    # Choose appropriate number of components
    n_components = min(2, n_samples - 1, n_features)
    print(f"Using {n_components} components for PCA with {n_samples} samples")
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    features_pca = pca.fit_transform(features)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    explained_variance_sum = sum(explained_variance)
    
    # Create DataFrame for plotting
    df = pd.DataFrame()
    
    # If we have 2D output, create normal scatter plot
    if n_components == 2:
        df['x'] = features_pca[:, 0]
        df['y'] = features_pca[:, 1]
    # If we only have 1D output, create artificial y-axis with small random noise
    elif n_components == 1:
        df['x'] = features_pca[:, 0]
        # Add small random noise for y-axis
        np.random.seed(RANDOM_SEED)
        df['y'] = np.random.normal(0, 0.01, size=n_samples)
        plt.ylabel('Random Jitter (for visualization only)')
    
    df['label'] = ['AI-Generated' if l == 1 else 'Real' for l in labels]
    df['video'] = video_names
    
    # Plot without text labels
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=100)
    
    # Add explained variance information
    plt.title(f'PCA Visualization of Laplacian Pyramid Temporal Features\nExplained Variance: {explained_variance_sum:.2f}')
    
    if n_components == 2:
        plt.xlabel(f'PC1 ({explained_variance[0]:.2f} variance explained)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2f} variance explained)')
    else:
        plt.xlabel(f'PC1 ({explained_variance[0]:.2f} variance explained)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # If enough samples, save feature importance information
    if n_samples >= 5 and feature_names:
        # Save feature loadings (PCA components)
        plt.figure(figsize=(14, 10))
        components = pca.components_
        
        # Plot the feature loadings
        if n_components == 2:
            # Create a DataFrame for the feature loadings
            loadings_df = pd.DataFrame({
                'Feature': feature_names,
                'PC1': components[0],
                'PC2': components[1]
            })
            
            # Sort by PC1 magnitude
            loadings_df = loadings_df.reindex(loadings_df['PC1'].abs().sort_values(ascending=False).index)
            
            # Plot top 10 features for PC1
            plt.subplot(1, 2, 1)
            sns.barplot(x='PC1', y='Feature', data=loadings_df.head(10))
            plt.title('Top 10 Features - PC1')
            
            # Sort by PC2 magnitude
            loadings_df = loadings_df.reindex(loadings_df['PC2'].abs().sort_values(ascending=False).index)
            
            # Plot top 10 features for PC2
            plt.subplot(1, 2, 2)
            sns.barplot(x='PC2', y='Feature', data=loadings_df.head(10))
            plt.title('Top 10 Features - PC2')
        else:
            # Just plot PC1 for 1D case
            loadings_df = pd.DataFrame({
                'Feature': feature_names,
                'PC1': components[0]
            })
            
            # Sort by PC1 magnitude
            loadings_df = loadings_df.reindex(loadings_df['PC1'].abs().sort_values(ascending=False).index)
            
            # Plot top 10 features for PC1
            sns.barplot(x='PC1', y='Feature', data=loadings_df.head(10))
            plt.title('Top 10 Features - PC1')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'feature_loadings.png'))
        plt.close()
    
    return pca

def train_and_evaluate_models(train_features, train_labels, test_features, test_labels, feature_names):
    """Train SVM and Logistic Regression classifiers and evaluate performance"""
    models = {}
    
    # Create SVM pipeline with standardization
    print("Training SVM classifier...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ])
    
    # Create Logistic Regression pipeline with standardization
    print("Training Logistic Regression classifier...")
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ])
    
    # Train SVM and make predictions
    svm_pipeline.fit(train_features, train_labels)
    svm_predictions = svm_pipeline.predict(test_features)
    
    # Print SVM classification report
    print("\nSVM Classification Report:")
    print(classification_report(test_labels, svm_predictions, target_names=['Real', 'AI-Generated']))
    
    # Plot SVM confusion matrix
    cm = confusion_matrix(test_labels, svm_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'AI-Generated'], yticklabels=['Real', 'AI-Generated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('SVM Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'svm_confusion_matrix.png'))
    plt.close()
    
    # Train Logistic Regression and make predictions
    lr_pipeline.fit(train_features, train_labels)
    lr_predictions = lr_pipeline.predict(test_features)
    
    # Print Logistic Regression classification report
    print("\nLogistic Regression Classification Report:")
    print(classification_report(test_labels, lr_predictions, target_names=['Real', 'AI-Generated']))
    
    # Plot Logistic Regression confusion matrix
    cm = confusion_matrix(test_labels, lr_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'AI-Generated'], yticklabels=['Real', 'AI-Generated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Logistic Regression Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'lr_confusion_matrix.png'))
    plt.close()
    
    # Store models and predictions
    models['svm'] = svm_pipeline
    models['lr'] = lr_pipeline
    models['svm_predictions'] = svm_predictions
    models['lr_predictions'] = lr_predictions
    
    return models

def analyze_prediction_errors(test_features, test_labels, predictions, video_names, model_name):
    """Analyze misclassified videos"""
    errors = test_labels != predictions
    error_indices = np.where(errors)[0]
    
    if len(error_indices) > 0:
        print(f"\nMisclassified Videos ({model_name}):")
        for idx in error_indices:
            true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
            pred_label = "Real" if predictions[idx] == 0 else "AI-Generated"
            print(f"Video: {video_names[idx]}, True: {true_label}, Predicted: {pred_label}")
    else:
        print(f"\nNo misclassified videos for {model_name}!")

def visualize_laplacian_examples(video_dirs, save_path, pyramid_levels=4):
    """Visualize example Laplacian pyramids from real and fake videos"""
    if not video_dirs:
        print("No video directories provided for visualization.")
        return
    
    # Select first video directory
    video_dir = video_dirs[0]
    
    # Find first frame
    frame_files = [f for f in os.listdir(video_dir) if f.startswith('frame_') and f.endswith(('.jpg', '.png'))]
    if not frame_files:
        print(f"No frames found in {video_dir}")
        return
    
    frame_files.sort()
    frame_path = os.path.join(video_dir, frame_files[0])
    
    # Read the frame
    image = cv2.imread(frame_path)
    if image is None:
        print(f"Could not read {frame_path}")
        return
    
    # Convert to grayscale and resize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    # Build Laplacian pyramid
    pyramid = build_laplacian_pyramid(gray, levels=pyramid_levels)
    
    # Set up plot
    n_levels = len(pyramid)
    fig, axes = plt.subplots(2, n_levels, figsize=(4*n_levels, 8))
    
    # Original grayscale image
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Empty the other plots in the first row
    for i in range(1, n_levels):
        axes[0, i].axis('off')
    
    # Plot each level of the pyramid
    for i, level in enumerate(pyramid):
        # For visualization, we need to normalize the Laplacian levels
        # Normal range is around -1 to 1, so we shift and scale
        vis_level = (level + 0.5) * 0.5  # Shift from [-0.5, 0.5] to [0, 1]
        
        axes[1, i].imshow(vis_level, cmap='gray')
        axes[1, i].set_title(f'Level {i+1}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Laplacian Pyramid Example\nVideo: {os.path.basename(video_dir)}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def create_results_summary(train_features, train_labels, test_features, test_labels, 
                           test_videos, models, feature_names):
    """Create and save a summary of results"""
    # Count real and fake videos in the training and test sets
    train_fake_count = sum(1 for label in train_labels if label == 1)
    train_real_count = sum(1 for label in train_labels if label == 0)
    test_fake_count = sum(1 for label in test_labels if label == 1)
    test_real_count = sum(1 for label in test_labels if label == 0)
    
    # Calculate total number of videos
    total_videos = len(train_features) + len(test_features)
    
    with open(os.path.join(RESULTS_DIR, 'results_summary.txt'), 'w') as f:
        f.write("DEEPFAKE DETECTION USING LAPLACIAN PYRAMID TEMPORAL ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training set: {len(train_features)} videos ({train_fake_count} fake, {train_real_count} real)\n")
        f.write(f"Test set: {len(test_features)} videos ({test_fake_count} fake, {test_real_count} real)\n")
        f.write(f"Total: {total_videos} videos\n\n")
        
        # SVM Results
        f.write("SVM Classification Report:\n")
        f.write(classification_report(test_labels, models['svm_predictions'], 
                                     target_names=['Real', 'AI-Generated']))
        
        # Add SVM misclassified videos
        f.write("\nSVM Misclassified Videos:\n")
        svm_errors = test_labels != models['svm_predictions']
        svm_error_indices = np.where(svm_errors)[0]
        
        if len(svm_error_indices) > 0:
            for idx in svm_error_indices:
                true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
                pred_label = "Real" if models['svm_predictions'][idx] == 0 else "AI-Generated"
                f.write(f"Video: {test_videos[idx]}, True: {true_label}, Predicted: {pred_label}\n")
        else:
            f.write("No misclassified videos for SVM!\n")
        
        # Logistic Regression Results
        f.write("\n\nLogistic Regression Classification Report:\n")
        f.write(classification_report(test_labels, models['lr_predictions'], 
                                     target_names=['Real', 'AI-Generated']))
        
        # Add LR misclassified videos
        f.write("\nLogistic Regression Misclassified Videos:\n")
        lr_errors = test_labels != models['lr_predictions']
        lr_error_indices = np.where(lr_errors)[0]
        
        if len(lr_error_indices) > 0:
            for idx in lr_error_indices:
                true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
                pred_label = "Real" if models['lr_predictions'][idx] == 0 else "AI-Generated"
                f.write(f"Video: {test_videos[idx]}, True: {true_label}, Predicted: {pred_label}\n")
        else:
            f.write("No misclassified videos for Logistic Regression!\n")
        
        # Model comparison
        f.write("\n\nModel Comparison:\n")
        svm_acc = (test_labels == models['svm_predictions']).mean()
        lr_acc = (test_labels == models['lr_predictions']).mean()
        f.write(f"SVM Accuracy: {svm_acc:.4f}\n")
        f.write(f"Logistic Regression Accuracy: {lr_acc:.4f}\n")
        
        best_model = "SVM" if svm_acc >= lr_acc else "Logistic Regression"
        f.write(f"Best Model: {best_model}\n")

def main():
    print("=" * 80)
    print("DEEPFAKE DETECTION USING LAPLACIAN PYRAMID TEMPORAL ANALYSIS")
    print("=" * 80)
    
    # Get video directories for training and testing with specified limits
    print("\nSelecting video directories...")
    
    # Get directories for training with exact limits
    train_fake_dirs = get_video_dirs(TRAIN_FAKE_PATH, limit=TRAIN_FAKE_LIMIT, seed=RANDOM_SEED)
    train_real_dirs = get_video_dirs(TRAIN_REAL_PATH, limit=TRAIN_REAL_LIMIT, seed=RANDOM_SEED)
    
    # Get directories for testing with exact limits
    test_fake_dirs = get_video_dirs(TEST_FAKE_PATH, limit=TEST_FAKE_LIMIT, seed=RANDOM_SEED)
    test_real_dirs = get_video_dirs(TEST_REAL_PATH, limit=TEST_REAL_LIMIT, seed=RANDOM_SEED)
    
    print(f"Training set: {len(train_fake_dirs)} fake, {len(train_real_dirs)} real videos")
    print(f"Test set: {len(test_fake_dirs)} fake, {len(test_real_dirs)} real videos")
    
    # Visualize example Laplacian pyramids
    if train_fake_dirs:
        visualize_laplacian_examples(train_fake_dirs, os.path.join(RESULTS_DIR, 'fake_laplacian_example.png'))
    if train_real_dirs:
        visualize_laplacian_examples(train_real_dirs, os.path.join(RESULTS_DIR, 'real_laplacian_example.png'))
    
    # Process videos
    print("\nExtracting features from training videos...")
    train_fake_features, train_fake_labels, train_fake_videos, feature_names = process_videos(train_fake_dirs, 'fake')
    train_real_features, train_real_labels, train_real_videos, _ = process_videos(train_real_dirs, 'real')
    
    # Combine training data
    has_train_fake = len(train_fake_features) > 0
    has_train_real = len(train_real_features) > 0
    
    if has_train_fake and has_train_real:
        train_features = np.vstack((train_fake_features, train_real_features))
        train_labels = np.concatenate((train_fake_labels, train_real_labels))
        train_videos = train_fake_videos + train_real_videos
    elif has_train_fake:
        train_features = train_fake_features
        train_labels = train_fake_labels
        train_videos = train_fake_videos
    elif has_train_real:
        train_features = train_real_features
        train_labels = train_real_labels
        train_videos = train_real_videos
    else:
        train_features = np.array([])
        train_labels = np.array([])
        train_videos = []
    
    # Process test videos
    print("\nExtracting features from test videos...")
    test_fake_features, test_fake_labels, test_fake_videos, _ = process_videos(test_fake_dirs, 'fake')
    test_real_features, test_real_labels, test_real_videos, _ = process_videos(test_real_dirs, 'real')
    
    # Combine test data
    has_test_fake = len(test_fake_features) > 0
    has_test_real = len(test_real_features) > 0
    
    if has_test_fake and has_test_real:
        test_features = np.vstack((test_fake_features, test_real_features))
        test_labels = np.concatenate((test_fake_labels, test_real_labels))
        test_videos = test_fake_videos + test_real_videos
    elif has_test_fake:
        test_features = test_fake_features
        test_labels = test_fake_labels
        test_videos = test_fake_videos
    elif has_test_real:
        test_features = test_real_features
        test_labels = test_real_labels
        test_videos = test_real_videos
    else:
        test_features = np.array([])
        test_labels = np.array([])
        test_videos = []
    
    # Combine all data for overall visualization
    if len(train_features) > 0 and len(test_features) > 0:
        all_features = np.vstack((train_features, test_features))
        all_labels = np.concatenate((train_labels, test_labels))
        all_videos = train_videos + test_videos
    elif len(train_features) > 0:
        all_features = train_features
        all_labels = train_labels
        all_videos = train_videos
    elif len(test_features) > 0:
        all_features = test_features
        all_labels = test_labels
        all_videos = test_videos
    else:
        all_features = np.array([])
        all_labels = np.array([])
        all_videos = []
    
    # Count real and fake videos in the training and test sets
    train_fake_count = sum(1 for label in train_labels if label == 1)
    train_real_count = sum(1 for label in train_labels if label == 0)
    test_fake_count = sum(1 for label in test_labels if label == 1)
    test_real_count = sum(1 for label in test_labels if label == 0)
    
    print(f"\nFeatures extracted:")
    print(f"Training set: {len(train_features)} videos ({train_fake_count} fake, {train_real_count} real)")
    print(f"Test set: {len(test_features)} videos ({test_fake_count} fake, {test_real_count} real)")
    print(f"Total: {len(all_features)} videos")
    
    if len(train_features) == 0 or len(test_features) == 0:
        print("Error: Not enough valid videos to continue analysis")
        return
    
    # Check if we have at least 3 examples of each class in both train and test sets
    if train_fake_count < 3 or train_real_count < 3 or test_fake_count < 3 or test_real_count < 3:
        print("Warning: Not enough samples for each class. For best results, ensure at least 3 samples of each class in both training and test sets.")
    
    # Save the feature names for reference
    if feature_names:
        with open(os.path.join(RESULTS_DIR, 'feature_names.txt'), 'w') as f:
            for i, name in enumerate(feature_names):
                f.write(f"{i+1}. {name}\n")
    
    # Generate visualizations
    print("\nGenerating t-SNE visualizations...")
    visualize_tsne(train_features, train_labels, train_videos, os.path.join(RESULTS_DIR, 'train_tsne.png'))
    visualize_tsne(test_features, test_labels, test_videos, os.path.join(RESULTS_DIR, 'test_tsne.png'))
    visualize_tsne(all_features, all_labels, all_videos, os.path.join(RESULTS_DIR, 'all_tsne.png'))
    
    print("\nGenerating PCA visualizations...")
    visualize_pca(train_features, train_labels, train_videos, os.path.join(RESULTS_DIR, 'train_pca.png'), feature_names)
    visualize_pca(test_features, test_labels, test_videos, os.path.join(RESULTS_DIR, 'test_pca.png'), feature_names)
    visualize_pca(all_features, all_labels, all_videos, os.path.join(RESULTS_DIR, 'all_pca.png'), feature_names)
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    models = train_and_evaluate_models(train_features, train_labels, test_features, test_labels, feature_names)
    
    # Analyze errors for each model
    analyze_prediction_errors(test_features, test_labels, models['svm_predictions'], test_videos, "SVM")
    analyze_prediction_errors(test_features, test_labels, models['lr_predictions'], test_videos, "Logistic Regression")
    
    # Save results summary
    create_results_summary(train_features, train_labels, test_features, test_labels, test_videos, models, feature_names)
    
    print(f"\nAnalysis complete! Results saved in the {RESULTS_DIR} directory.")

if __name__ == "__main__":
    main()