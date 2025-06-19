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
RESULTS_DIR = "ml_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Modified limits as per request
TRAIN_FAKE_LIMIT = 25  # Limit to 50 fake videos for training
TRAIN_REAL_LIMIT = 25  # Limit to 50 real videos for training
TEST_FAKE_LIMIT = 5   # Limit to 10 fake videos for testing
TEST_REAL_LIMIT = 5   # Limit to 10 real videos for testing

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

def compute_hog_features(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Compute HOG features for a single image"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to a fixed size to ensure consistency and avoid HOG dimension errors
    gray = cv2.resize(gray, (128, 128))
    
    # Make sure dimensions satisfy HOG requirements
    # The window size - block size must be divisible by block stride
    # Use standard parameters that are known to work well
    win_size = (128, 128)
    cell_size = pixels_per_cell
    block_size = (cells_per_block[0] * cell_size[0], cells_per_block[1] * cell_size[1])
    block_stride = cell_size  # Same as cell size for standard HOG
    
    # Create HOG descriptor
    hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, orientations)
    hog_features = hog.compute(gray)
    
    return hog_features.flatten()

def extract_hog_temporal_features(video_dir):
    """Extract HOG features from all frames in a video directory and compute temporal features"""
    frame_files = [f for f in os.listdir(video_dir) if f.startswith('frame_') and f.endswith(('.jpg', '.png'))]
    frame_files.sort()  # Ensure frames are in order
    
    # Skip if too few frames
    if len(frame_files) < 3:
        print(f"Warning: Too few frames in {video_dir} - skipping")
        return None
    
    # Load and process all frames
    hog_features_list = []
    
    for frame_file in frame_files:
        try:
            frame_path = os.path.join(video_dir, frame_file)
            image = cv2.imread(frame_path)
            
            if image is None:
                print(f"Warning: Could not read {frame_path} - skipping")
                continue
                
            # Extract HOG features
            hog_features = compute_hog_features(image)
            hog_features_list.append(hog_features)
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")
            continue
    
    # Convert to numpy array for easier computation
    hog_features_array = np.array(hog_features_list)
    
    # Skip if too few frames were successfully processed
    if len(hog_features_array) < 3:
        print(f"Warning: Too few valid frames in {video_dir} - skipping")
        return None
    
    # Calculate frame-to-frame differences (temporal information)
    hog_feature_diffs = np.abs(hog_features_array[1:] - hog_features_array[:-1])
    
    # Initialize feature dictionary
    features = {}
    
    # 1. Basic HOG Statistics
    features['hog_mean'] = np.mean(hog_features_array)
    features['hog_std'] = np.std(hog_features_array)
    features['hog_median'] = np.median(hog_features_array)
    features['hog_min'] = np.min(hog_features_array)
    features['hog_max'] = np.max(hog_features_array)
    
    # 2. Temporal Variation Metrics
    features['hog_temporal_std'] = np.std(hog_features_array, axis=0).mean()  # Average std across time
    features['hog_temporal_mean_diff'] = np.mean(hog_feature_diffs)  # Average difference between frames
    features['hog_temporal_max_diff'] = np.max(hog_feature_diffs)  # Maximum difference
    features['hog_temporal_diff_std'] = np.std(hog_feature_diffs)  # Std of differences
    
    # 3. Consistency Metrics
    # Higher values may indicate real videos (more stable)
    features['hog_stability'] = 1.0 / (features['hog_temporal_std'] + 1e-6)
    
    # Measure of consistency across HOG cells (spatial uniformity)
    features['hog_uniformity'] = 1.0 / (np.std(np.mean(hog_features_array, axis=0)) + 1e-6)
    
    # Measure of consistency between adjacent frames
    consistency_scores = []
    for i in range(len(hog_features_array) - 1):
        corr = pearsonr(hog_features_array[i], hog_features_array[i+1])[0]
        if not np.isnan(corr):
            consistency_scores.append(corr)
    
    if consistency_scores:
        features['hog_temporal_consistency'] = np.mean(consistency_scores)
        features['hog_temporal_consistency_std'] = np.std(consistency_scores)
    else:
        features['hog_temporal_consistency'] = 0
        features['hog_temporal_consistency_std'] = 0
    
    # 4. Spectral Analysis (if enough frames)
    if len(hog_features_array) >= 8:  # Need minimum number of frames for FFT
        # Compute FFT for each HOG dimension and average
        fft_magnitudes = []
        for i in range(hog_features_array.shape[1]):
            feature_series = hog_features_array[:, i]
            # Remove mean (DC component)
            feature_series = feature_series - np.mean(feature_series)
            
            # Compute FFT
            fft = np.abs(np.fft.rfft(feature_series))
            fft_magnitudes.append(fft)
        
        # Average FFT across all HOG dimensions
        avg_fft = np.mean(fft_magnitudes, axis=0)
        
        # Calculate energy in different frequency bands
        n_bands = 3  # Low, mid, high
        band_size = len(avg_fft) // n_bands
        
        total_energy = np.sum(avg_fft)
        if total_energy > 0:  # Avoid division by zero
            features['hog_low_freq_energy'] = np.sum(avg_fft[:band_size]) / total_energy
            features['hog_mid_freq_energy'] = np.sum(avg_fft[band_size:2*band_size]) / total_energy
            features['hog_high_freq_energy'] = np.sum(avg_fft[2*band_size:]) / total_energy
            
            # Peak frequency information
            peak_freq_idx = np.argmax(avg_fft)
            features['hog_peak_freq_idx'] = peak_freq_idx
            features['hog_peak_freq_magnitude'] = avg_fft[peak_freq_idx] / total_energy
        else:
            features['hog_low_freq_energy'] = 0
            features['hog_mid_freq_energy'] = 0
            features['hog_high_freq_energy'] = 0
            features['hog_peak_freq_idx'] = 0
            features['hog_peak_freq_magnitude'] = 0
    else:
        # Not enough frames for spectral analysis
        features['hog_low_freq_energy'] = 0
        features['hog_mid_freq_energy'] = 0
        features['hog_high_freq_energy'] = 0
        features['hog_peak_freq_idx'] = 0
        features['hog_peak_freq_magnitude'] = 0
    
    # 5. Eigenvalue Analysis (if enough samples)
    if hog_features_array.shape[0] > 3 and hog_features_array.shape[1] > 3:
        try:
            # Center the data
            centered_hog = hog_features_array - np.mean(hog_features_array, axis=0)
            
            # Calculate covariance matrix
            cov_matrix = np.cov(centered_hog, rowvar=False)
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort in descending order
            
            # Only use positive eigenvalues
            eigenvalues = eigenvalues[eigenvalues > 0]
            
            if len(eigenvalues) > 0:
                total_variance = np.sum(eigenvalues)
                
                # Ratio of largest eigenvalue to sum (principal mode)
                features['hog_top_eigenvalue_ratio'] = eigenvalues[0] / total_variance if total_variance > 0 else 0
                
                # Ratio of top 3 eigenvalues (or fewer if not enough)
                top_n = min(3, len(eigenvalues))
                features['hog_top3_eigenvalue_ratio'] = np.sum(eigenvalues[:top_n]) / total_variance if total_variance > 0 else 0
                
                # Ratio between first and second eigenvalues (if available)
                if len(eigenvalues) > 1:
                    features['hog_eigenvalue_decay'] = eigenvalues[0] / eigenvalues[1]
                else:
                    features['hog_eigenvalue_decay'] = 1.0
            else:
                features['hog_top_eigenvalue_ratio'] = 0
                features['hog_top3_eigenvalue_ratio'] = 0
                features['hog_eigenvalue_decay'] = 1.0
        except (np.linalg.LinAlgError, ValueError) as e:
            # Handle potential numerical issues
            print(f"Eigenvalue calculation error in {video_dir}: {e}")
            features['hog_top_eigenvalue_ratio'] = 0
            features['hog_top3_eigenvalue_ratio'] = 0
            features['hog_eigenvalue_decay'] = 1.0
    else:
        features['hog_top_eigenvalue_ratio'] = 0
        features['hog_top3_eigenvalue_ratio'] = 0
        features['hog_eigenvalue_decay'] = 1.0
    
    # Return all features as numpy array
    return np.array(list(features.values())), list(features.keys())

def process_videos(video_dirs, label):
    """Process a list of video directories and extract features"""
    features_list = []
    labels = []
    video_names = []
    feature_names = None
    
    for video_dir in tqdm(video_dirs, desc=f"Processing {label} videos"):
        try:
            video_name = os.path.basename(video_dir)
            
            # Extract HOG features and temporal analysis
            result = extract_hog_temporal_features(video_dir)
            
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
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=100)
    
    # No labels for individual points to keep visualization clean
    
    plt.title('t-SNE Visualization of HOG Temporal Features')
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
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=100)
    
    # No labels for individual points to keep visualization clean
    
    # Add explained variance information
    plt.title(f'PCA Visualization of HOG Temporal Features\nExplained Variance: {explained_variance_sum:.2f}')
    
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
    results = {}
    
    # 1. Train and evaluate SVM
    print("\nTraining SVM classifier...")
    svm_clf = SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED)
    svm_clf.fit(train_features, train_labels)
    svm_predictions = svm_clf.predict(test_features)
    
    # Print SVM classification report
    print("\nSVM Classification Report:")
    svm_report = classification_report(test_labels, svm_predictions, target_names=['Real', 'AI-Generated'])
    print(svm_report)
    
    # Plot SVM confusion matrix
    svm_cm = confusion_matrix(test_labels, svm_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'AI-Generated'], yticklabels=['Real', 'AI-Generated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('SVM Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'svm_confusion_matrix.png'))
    plt.close()
    
    # 2. Train and evaluate Logistic Regression
    print("\nTraining Logistic Regression classifier...")
    lr_clf = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    lr_clf.fit(train_features, train_labels)
    lr_predictions = lr_clf.predict(test_features)
    
    # Print Logistic Regression classification report
    print("\nLogistic Regression Classification Report:")
    lr_report = classification_report(test_labels, lr_predictions, target_names=['Real', 'AI-Generated'])
    print(lr_report)
    
    # Plot Logistic Regression confusion matrix
    lr_cm = confusion_matrix(test_labels, lr_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'AI-Generated'], yticklabels=['Real', 'AI-Generated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Logistic Regression Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'lr_confusion_matrix.png'))
    plt.close()
    
    # Feature importance for Logistic Regression
    if feature_names:
        plt.figure(figsize=(12, 8))
        
        # Sort coefficients by absolute value
        coefficients = lr_clf.coef_[0]
        importance = np.abs(coefficients)
        indices = np.argsort(importance)[::-1]
        
        plt.bar(range(len(coefficients)), coefficients[indices])
        plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Coefficient')
        plt.title('Logistic Regression Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'lr_feature_importance.png'))
        plt.close()
    
    results['svm'] = {
        'classifier': svm_clf,
        'predictions': svm_predictions,
        'report': svm_report
    }
    
    results['lr'] = {
        'classifier': lr_clf,
        'predictions': lr_predictions, 
        'report': lr_report
    }
    
    return results

def analyze_prediction_errors(test_features, test_labels, predictions, video_names, model_name):
    """Analyze misclassified videos"""
    errors = test_labels != predictions
    error_indices = np.where(errors)[0]
    
    if len(error_indices) > 0:
        print(f"\nMisclassified Videos by {model_name}:")
        for idx in error_indices:
            true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
            pred_label = "Real" if predictions[idx] == 0 else "AI-Generated"
            print(f"Video: {video_names[idx]}, True: {true_label}, Predicted: {pred_label}")
        
        return error_indices
    
    return []

def main():
    print("=" * 80)
    print("DEEPFAKE DETECTION USING HOG TEMPORAL ANALYSIS")
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
    model_results = train_and_evaluate_models(train_features, train_labels, test_features, test_labels, feature_names)
    
    # Analyze errors for each model
    svm_predictions = model_results['svm']['predictions']
    lr_predictions = model_results['lr']['predictions']
    
    svm_errors = analyze_prediction_errors(test_features, test_labels, svm_predictions, test_videos, "SVM")
    lr_errors = analyze_prediction_errors(test_features, test_labels, lr_predictions, test_videos, "Logistic Regression")
    
    # Find common errors between models
    common_errors = set(svm_errors).intersection(set(lr_errors))
    if common_errors:
        print("\nVideos misclassified by both models:")
        for idx in common_errors:
            true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
            print(f"Video: {test_videos[idx]}, True: {true_label}")
    
    # Save results summary
    with open(os.path.join(RESULTS_DIR, 'results_summary.txt'), 'w') as f:
        f.write("DEEPFAKE DETECTION USING HOG TEMPORAL ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training set: {len(train_features)} videos ({train_fake_count} fake, {train_real_count} real)\n")
        f.write(f"Test set: {len(test_features)} videos ({test_fake_count} fake, {test_real_count} real)\n")
        f.write(f"Total: {len(all_features)} videos\n\n")
        
        f.write("SVM Classification Report:\n")
        f.write(model_results['svm']['report'] + "\n\n")
        
        f.write("Logistic Regression Classification Report:\n")
        f.write(model_results['lr']['report'] + "\n\n")
        
        # Add misclassified videos by SVM
        f.write("Misclassified Videos by SVM:\n")
        for idx in svm_errors:
            true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
            pred_label = "Real" if svm_predictions[idx] == 0 else "AI-Generated"
            f.write(f"Video: {test_videos[idx]}, True: {true_label}, Predicted: {pred_label}\n")
        
        # Add misclassified videos by Logistic Regression
        f.write("\nMisclassified Videos by Logistic Regression:\n")
        for idx in lr_errors:
            true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
            pred_label = "Real" if lr_predictions[idx] == 0 else "AI-Generated"
            f.write(f"Video: {test_videos[idx]}, True: {true_label}, Predicted: {pred_label}\n")
        
        # Add common errors
        f.write("\nVideos misclassified by both models:\n")
        for idx in common_errors:
            true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
            f.write(f"Video: {test_videos[idx]}, True: {true_label}\n")
    
    # Compare model performance with a side-by-side bar chart
    plt.figure(figsize=(10, 6))
    
    # Calculate accuracy, precision, recall, and F1 score for each model
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Accuracy': [
            accuracy_score(test_labels, svm_predictions),
            accuracy_score(test_labels, lr_predictions)
        ],
        'Precision': [
            precision_score(test_labels, svm_predictions, pos_label=1),
            precision_score(test_labels, lr_predictions, pos_label=1)
        ],
        'Recall': [
            recall_score(test_labels, svm_predictions, pos_label=1),
            recall_score(test_labels, lr_predictions, pos_label=1)
        ],
        'F1 Score': [
            f1_score(test_labels, svm_predictions, pos_label=1),
            f1_score(test_labels, lr_predictions, pos_label=1)
        ]
    }
    
    # Create a DataFrame for plotting
    metrics_df = pd.DataFrame(metrics, index=['SVM', 'Logistic Regression'])
    
    # Plot as a grouped bar chart
    metrics_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'))
    plt.close()
    
    print(f"\nAnalysis complete! Results saved in the {RESULTS_DIR} directory.")

if __name__ == "__main__":
    main()