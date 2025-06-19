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
from scipy import signal, stats
from scipy.stats import pearsonr
from collections import Counter
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
RESULTS_DIR = "fourier_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Modified limits
TRAIN_FAKE_LIMIT = 250  # Limit to 25 fake videos for training
TRAIN_REAL_LIMIT = 250  # Limit to 25 real videos for training
TEST_FAKE_LIMIT = 60    # Limit to 6 fake videos for testing
TEST_REAL_LIMIT = 60    # Limit to 6 real videos for testing

# Fourier analysis parameters
NUM_RINGS = 5       # Number of concentric rings in frequency domain
NUM_SECTORS = 4     # Number of angular sectors in frequency domain

def get_video_dirs(directory, limit=None, seed=RANDOM_SEED, exclude_dirs=None):
    """Get a specified number of video directories from a parent directory
    
    Args:
        directory: Directory containing video subdirectories
        limit: Maximum number of directories to return
        seed: Random seed for consistent selection
        exclude_dirs: List of directory paths to exclude
    """
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return []
        
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

def compute_fourier_features(image):
    """Compute Fourier transform features for a single image
    
    Args:
        image: Input image
        
    Returns:
        Array of features extracted from the Fourier transform
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize to a fixed size to ensure consistency
    gray = cv2.resize(gray, (128, 128))
    
    # Convert to float for FFT
    gray_float = gray.astype(np.float32) / 255.0
    
    # Apply 2D FFT
    f_transform = np.fft.fft2(gray_float)
    
    # Shift zero frequency component to center
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Compute magnitude spectrum (log scale for better visualization)
    magnitude_spectrum = np.log(1 + np.abs(f_transform_shifted))
    
    # Compute phase spectrum
    phase_spectrum = np.angle(f_transform_shifted)
    
    # Extract features from concentric rings in the frequency domain
    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2
    
    # Create distance map from center
    y_grid, x_grid = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    
    # Define rings for frequency bands (logarithmically spaced)
    max_radius = min(center_y, center_x)
    
    # Using logarithmic spacing for rings to focus more on low frequencies
    log_space = np.logspace(0, np.log10(max_radius), NUM_RINGS + 1)
    
    # Features to store
    ring_features = []
    
    # Extract statistics from each ring
    for i in range(NUM_RINGS):
        inner_radius = log_space[i]
        outer_radius = log_space[i + 1]
        
        # Create mask for this ring
        ring_mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)
        
        # Extract magnitude values in this ring
        ring_magnitudes = magnitude_spectrum[ring_mask]
        
        if len(ring_magnitudes) > 0:
            # Compute statistics for magnitudes
            ring_mean = np.mean(ring_magnitudes)
            ring_std = np.std(ring_magnitudes)
            ring_max = np.max(ring_magnitudes)
            ring_median = np.median(ring_magnitudes)
            ring_energy = np.sum(ring_magnitudes**2)
            
            # Extract phase values in this ring
            ring_phases = phase_spectrum[ring_mask]
            
            # Compute phase statistics
            phase_mean = np.mean(ring_phases)
            phase_std = np.std(ring_phases)
            
            # Add features for this ring
            ring_features.extend([
                ring_mean, ring_std, ring_max, ring_median, ring_energy,
                phase_mean, phase_std
            ])
        else:
            # Placeholder values if the ring is empty
            ring_features.extend([0, 0, 0, 0, 0, 0, 0])
    
    # Compute directional frequency features
    # Define angular sectors
    sector_size = 2 * np.pi / NUM_SECTORS
    
    # Calculate angles from center
    angles = np.arctan2(y_grid - center_y, x_grid - center_x) + np.pi  # Shift to [0, 2π]
    
    # Directional features
    directional_features = []
    
    for i in range(NUM_SECTORS):
        sector_start = i * sector_size
        sector_end = (i + 1) * sector_size
        
        # Create mask for this sector
        sector_mask = (angles >= sector_start) & (angles < sector_end)
        
        # Extract magnitude values in this sector
        sector_magnitudes = magnitude_spectrum[sector_mask]
        
        if len(sector_magnitudes) > 0:
            # Compute statistics
            sector_mean = np.mean(sector_magnitudes)
            sector_std = np.std(sector_magnitudes)
            sector_energy = np.sum(sector_magnitudes**2)
            
            directional_features.extend([sector_mean, sector_std, sector_energy])
        else:
            directional_features.extend([0, 0, 0])
    
    # Compute global spectral features
    # DCT-like coefficient analysis (focusing on low-frequency components)
    low_freq_region = magnitude_spectrum[
        max(0, center_y-10):min(h, center_y+10), 
        max(0, center_x-10):min(w, center_x+10)
    ]
    
    low_freq_mean = np.mean(low_freq_region) if low_freq_region.size > 0 else 0
    low_freq_std = np.std(low_freq_region) if low_freq_region.size > 0 else 0
    low_freq_max = np.max(low_freq_region) if low_freq_region.size > 0 else 0
    
    # High frequency region (corners of the spectrum)
    high_freq_mask = distance_from_center > 0.7 * max_radius
    high_freq_values = magnitude_spectrum[high_freq_mask]
    high_freq_mean = np.mean(high_freq_values) if len(high_freq_values) > 0 else 0
    high_freq_std = np.std(high_freq_values) if len(high_freq_values) > 0 else 0
    
    # Ratio of high to low frequency energy (can detect sharpening/blurring)
    high_low_ratio = high_freq_mean / low_freq_mean if low_freq_mean > 0 else 0
    
    # Add global features
    global_features = [
        low_freq_mean, low_freq_std, low_freq_max,
        high_freq_mean, high_freq_std, high_low_ratio
    ]
    
    # Spectral texture descriptors - simplified
    mean_magnitude = np.mean(magnitude_spectrum)
    spectral_variance = np.var(magnitude_spectrum)
    
    texture_features = [mean_magnitude, spectral_variance]
    
    # Combine all features
    all_features = np.concatenate([
        ring_features, 
        directional_features,
        global_features,
        texture_features
    ])
    
    return all_features

def get_fourier_feature_names():
    """Generate descriptive names for the Fourier features"""
    feature_names = []
    
    # Ring features
    for i in range(NUM_RINGS):
        # Statistical features
        feature_names.extend([
            f'ring{i+1}_mean', 
            f'ring{i+1}_std', 
            f'ring{i+1}_max', 
            f'ring{i+1}_median', 
            f'ring{i+1}_energy',
            f'ring{i+1}_phase_mean', 
            f'ring{i+1}_phase_std'
        ])
    
    # Directional features
    for i in range(NUM_SECTORS):
        angle = i * (360 // NUM_SECTORS)
        feature_names.extend([
            f'dir{angle}deg_mean', 
            f'dir{angle}deg_std', 
            f'dir{angle}deg_energy'
        ])
    
    # Global spectral features
    feature_names.extend([
        'low_freq_mean', 
        'low_freq_std', 
        'low_freq_max',
        'high_freq_mean', 
        'high_freq_std', 
        'high_low_ratio'
    ])
    
    # Texture features
    feature_names.extend([
        'mean_magnitude', 
        'spectral_variance'
    ])
    
    return feature_names

def extract_frame_fourier_features(video_dir):
    """Extract Fourier features from all frames in a video directory
    
    Args:
        video_dir: Path to directory containing video frames
        
    Returns:
        Numpy array of Fourier features for each frame, or None if processing failed
    """
    # Find all frame files
    frame_files = [f for f in os.listdir(video_dir) if f.startswith('frame_') and f.endswith(('.jpg', '.png'))]
    frame_files.sort()  # Ensure frames are in order
    
    # Skip if too few frames
    if len(frame_files) < 3:
        print(f"Warning: Too few frames in {video_dir} - skipping")
        return None
    
    # Load and process all frames
    fourier_features_list = []
    
    for frame_file in frame_files:
        try:
            frame_path = os.path.join(video_dir, frame_file)
            image = cv2.imread(frame_path)
            
            if image is None:
                continue
                
            # Extract Fourier features
            fourier_features = compute_fourier_features(image)
            fourier_features_list.append(fourier_features)
        except Exception as e:
            print(f"Error processing frame {frame_path}: {e}")
            continue
    
    # Check if we successfully processed any frames
    if not fourier_features_list:
        return None
    
    # Convert to numpy array
    fourier_features_array = np.array(fourier_features_list)
    
    # Skip if too few frames were successfully processed
    if len(fourier_features_array) < 3:
        return None
    
    return fourier_features_array

def compute_temporal_features(features_array):
    """Compute temporal features from a sequence of frame features
    
    Args:
        features_array: Numpy array of frame features with shape (num_frames, num_features)
        
    Returns:
        Dictionary of temporal features
    """
    # Calculate frame-to-frame differences
    feature_diffs = np.abs(features_array[1:] - features_array[:-1])
    
    # Initialize feature dictionary
    features = {}
    
    # 1. Basic Statistical Features
    features['fourier_mean'] = np.mean(features_array)
    features['fourier_std'] = np.std(features_array)
    features['fourier_median'] = np.median(features_array)
    features['fourier_min'] = np.min(features_array)
    features['fourier_max'] = np.max(features_array)
    
    # 2. Temporal Variation Metrics
    features['fourier_temporal_std'] = np.std(features_array, axis=0).mean()
    features['fourier_temporal_mean_diff'] = np.mean(feature_diffs)
    features['fourier_temporal_max_diff'] = np.max(feature_diffs)
    features['fourier_temporal_diff_std'] = np.std(feature_diffs)
    
    # 3. Consistency Metrics
    features['fourier_stability'] = 1.0 / (features['fourier_temporal_std'] + 1e-6)
    features['fourier_uniformity'] = 1.0 / (np.std(np.mean(features_array, axis=0)) + 1e-6)
    
    # Measure of consistency between adjacent frames
    consistency_scores = []
    for i in range(len(features_array) - 1):
        corr = pearsonr(features_array[i], features_array[i+1])[0]
        if not np.isnan(corr):
            consistency_scores.append(corr)
    
    if consistency_scores:
        features['fourier_temporal_consistency'] = np.mean(consistency_scores)
        features['fourier_temporal_consistency_std'] = np.std(consistency_scores)
    else:
        features['fourier_temporal_consistency'] = 0
        features['fourier_temporal_consistency_std'] = 0
    
    # 4. Spectral Analysis of the temporal signal
    if len(features_array) >= 8:
        # Compute FFT for each Fourier dimension across time and average
        temporal_fft_magnitudes = []
        for i in range(features_array.shape[1]):
            feature_series = features_array[:, i]
            # Remove mean (DC component)
            feature_series = feature_series - np.mean(feature_series)
            
            # Compute FFT of the temporal signal
            fft = np.abs(np.fft.rfft(feature_series))
            temporal_fft_magnitudes.append(fft)
        
        # Average FFT across all Fourier dimensions
        avg_temporal_fft = np.mean(temporal_fft_magnitudes, axis=0)
        
        # Calculate energy in different frequency bands
        n_bands = 3  # Low, mid, high
        band_size = len(avg_temporal_fft) // n_bands
        
        if band_size > 0:  # Ensure we can divide into bands
            total_energy = np.sum(avg_temporal_fft)
            if total_energy > 0:
                features['temporal_low_freq_energy'] = np.sum(avg_temporal_fft[:band_size]) / total_energy
                features['temporal_mid_freq_energy'] = np.sum(avg_temporal_fft[band_size:2*band_size]) / total_energy
                features['temporal_high_freq_energy'] = np.sum(avg_temporal_fft[2*band_size:]) / total_energy
                
                # Peak frequency information
                peak_freq_idx = np.argmax(avg_temporal_fft)
                features['temporal_peak_freq_idx'] = peak_freq_idx
                features['temporal_peak_freq_magnitude'] = avg_temporal_fft[peak_freq_idx] / total_energy
            else:
                features['temporal_low_freq_energy'] = 0
                features['temporal_mid_freq_energy'] = 0
                features['temporal_high_freq_energy'] = 0
                features['temporal_peak_freq_idx'] = 0
                features['temporal_peak_freq_magnitude'] = 0
        else:
            features['temporal_low_freq_energy'] = 0
            features['temporal_mid_freq_energy'] = 0
            features['temporal_high_freq_energy'] = 0
            features['temporal_peak_freq_idx'] = 0
            features['temporal_peak_freq_magnitude'] = 0
    else:
        features['temporal_low_freq_energy'] = 0
        features['temporal_mid_freq_energy'] = 0
        features['temporal_high_freq_energy'] = 0
        features['temporal_peak_freq_idx'] = 0
        features['temporal_peak_freq_magnitude'] = 0
    
    # 5. Eigenvalue Analysis
    if features_array.shape[0] > 3 and features_array.shape[1] > 3:
        try:
            # Center the data
            centered_features = features_array - np.mean(features_array, axis=0)
            
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
                features['fourier_top_eigenvalue_ratio'] = eigenvalues[0] / total_variance if total_variance > 0 else 0
                
                # Ratio of top 3 eigenvalues (or fewer if not enough)
                top_n = min(3, len(eigenvalues))
                features['fourier_top3_eigenvalue_ratio'] = np.sum(eigenvalues[:top_n]) / total_variance if total_variance > 0 else 0
                
                # Ratio between first and second eigenvalues (if available)
                if len(eigenvalues) > 1:
                    features['fourier_eigenvalue_decay'] = eigenvalues[0] / eigenvalues[1]
                else:
                    features['fourier_eigenvalue_decay'] = 1.0
            else:
                features['fourier_top_eigenvalue_ratio'] = 0
                features['fourier_top3_eigenvalue_ratio'] = 0
                features['fourier_eigenvalue_decay'] = 1.0
        except (np.linalg.LinAlgError, ValueError):
            features['fourier_top_eigenvalue_ratio'] = 0
            features['fourier_top3_eigenvalue_ratio'] = 0
            features['fourier_eigenvalue_decay'] = 1.0
    else:
        features['fourier_top_eigenvalue_ratio'] = 0
        features['fourier_top3_eigenvalue_ratio'] = 0
        features['fourier_eigenvalue_decay'] = 1.0
    
    return features

def process_video(video_dir):
    """Process a single video directory and extract temporal Fourier features
    
    Args:
        video_dir: Path to directory containing video frames
        
    Returns:
        Tuple of (features_array, feature_names) or (None, None) if processing failed
    """
    try:
        # Extract frame features
        frame_features = extract_frame_fourier_features(video_dir)
        
        if frame_features is None:
            return None, None
        
        # Compute temporal features
        temporal_features = compute_temporal_features(frame_features)
        
        # Get feature names and values in consistent order
        feature_names = list(temporal_features.keys())
        feature_values = np.array([temporal_features[name] for name in feature_names])
        
        return feature_values, feature_names
    
    except Exception as e:
        print(f"Error processing video {video_dir}: {e}")
        return None, None

def process_videos(video_dirs, label):
    """Process a list of video directories and extract features
    
    Args:
        video_dirs: List of paths to video directories
        label: Label for the videos ('fake' or 'real')
        
    Returns:
        Tuple of (features_array, labels_array, video_names, feature_names)
    """
    features_list = []
    labels = []
    video_names = []
    all_feature_names = None
    
    for video_dir in tqdm(video_dirs, desc=f"Processing {label} videos"):
        try:
            video_name = os.path.basename(video_dir)
            
            # Process the video
            features, feature_names = process_video(video_dir)
            
            if features is not None:
                # Store the first set of feature names
                if all_feature_names is None:
                    all_feature_names = feature_names
                
                # Ensure the feature dimensions match
                if len(features) == len(all_feature_names):
                    features_list.append(features)
                    labels.append(1 if label == 'fake' else 0)  # 1 for fake, 0 for real
                    video_names.append(video_name)
                else:
                    print(f"Warning: Inconsistent feature dimensions for {video_dir}.")
                    print(f"Expected {len(all_feature_names)}, got {len(features)}.")
        except Exception as e:
            print(f"Error processing {video_dir}: {e}")
    
    if not features_list:
        return np.array([]), np.array([]), [], []
    
    # Convert to numpy arrays
    features_array = np.array(features_list)
    labels_array = np.array(labels)
    
    return features_array, labels_array, video_names, all_feature_names

def visualize_fourier_example(image_path, save_path):
    """Visualize the Fourier transform of an image
    
    Args:
        image_path: Path to the image file
        save_path: Path to save the visualization
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    
    # Calculate magnitude spectrum (log scale for better visualization)
    magnitude = np.log(1 + np.abs(f_shift))
    
    # Calculate phase spectrum
    phase = np.angle(f_shift)
    
    # Set up plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Magnitude spectrum
    axes[1].imshow(magnitude, cmap='viridis')
    axes[1].set_title('Magnitude Spectrum (Log Scale)')
    axes[1].axis('off')
    
    # Phase spectrum
    axes[2].imshow(phase, cmap='hsv')
    axes[2].set_title('Phase Spectrum')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_tsne(features, labels, video_names, save_path):
    """Apply t-SNE and visualize the results
    
    Args:
        features: Feature matrix
        labels: Labels array
        video_names: List of video names
        save_path: Path to save the visualization
    """
    # Handle case with small number of samples
    n_samples = features.shape[0]
    
    if n_samples <= 5:
        print(f"Warning: Not enough samples ({n_samples}) for t-SNE visualization. Skipping.")
        return
    
    # Set perplexity to min(30, n_samples/3) with a minimum of 2
    perplexity = min(30, max(2, n_samples // 3))
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, 
                n_iter=2000, random_state=RANDOM_SEED)
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
    
    plt.title('t-SNE Visualization of Fourier Temporal Features')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_pca(features, labels, video_names, save_path, feature_names):
    """Apply PCA and visualize the results
    
    Args:
        features: Feature matrix
        labels: Labels array
        video_names: List of video names
        save_path: Path to save the visualization
        feature_names: List of feature names
    """
    # Handle case with small number of samples
    n_samples = features.shape[0]
    n_features = features.shape[1]
    
    if n_samples < 2:
        print(f"Warning: Not enough samples ({n_samples}) for PCA visualization. Skipping.")
        return
    
    # Choose appropriate number of components
    n_components = min(2, n_samples - 1, n_features)
    
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
    
    # Plot (without text labels)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=100)
    
    # Add explained variance information
    plt.title(f'PCA Visualization of Fourier Temporal Features\nExplained Variance: {explained_variance_sum:.2f}')
    
    if n_components == 2:
        plt.xlabel(f'PC1 ({explained_variance[0]:.2f} variance explained)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2f} variance explained)')
    else:
        plt.xlabel(f'PC1 ({explained_variance[0]:.2f} variance explained)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def train_and_evaluate_model(train_features, train_labels, test_features, test_labels, feature_names):
    """Train SVM and Logistic Regression classifiers and evaluate performance
    
    Args:
        train_features: Training feature matrix
        train_labels: Training labels
        test_features: Test feature matrix
        test_labels: Test labels
        feature_names: List of feature names
        
    Returns:
        Dictionary containing trained models
    """
    models = {}
    
    # Create SVM pipeline with standardization
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ])
    
    # Create Logistic Regression pipeline with standardization
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ])
    
    # Train and evaluate SVM
    print("\nTraining SVM classifier...")
    svm_pipeline.fit(train_features, train_labels)
    svm_predictions = svm_pipeline.predict(test_features)
    
    print("\nSVM Classification Report:")
    print(classification_report(test_labels, svm_predictions, target_names=['Real', 'AI-Generated']))
    
    # Plot SVM confusion matrix
    svm_cm = confusion_matrix(test_labels, svm_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'AI-Generated'], 
                yticklabels=['Real', 'AI-Generated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('SVM Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'svm_confusion_matrix.png'))
    plt.close()
    
    # Train and evaluate Logistic Regression
    print("\nTraining Logistic Regression classifier...")
    lr_pipeline.fit(train_features, train_labels)
    lr_predictions = lr_pipeline.predict(test_features)
    
    print("\nLogistic Regression Classification Report:")
    print(classification_report(test_labels, lr_predictions, target_names=['Real', 'AI-Generated']))
    
    # Plot Logistic Regression confusion matrix
    lr_cm = confusion_matrix(test_labels, lr_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'AI-Generated'], 
                yticklabels=['Real', 'AI-Generated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Logistic Regression Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'lr_confusion_matrix.png'))
    plt.close()
    
    # Store models
    models['svm'] = svm_pipeline
    models['lr'] = lr_pipeline
    models['svm_predictions'] = svm_predictions
    models['lr_predictions'] = lr_predictions
    
    return models

def analyze_prediction_errors(test_features, test_labels, predictions, video_names, model_name):
    """Analyze misclassified videos
    
    Args:
        test_features: Test feature matrix
        test_labels: Test labels
        predictions: Model predictions
        video_names: List of video names
        model_name: Name of the model
    """
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

def create_results_summary(train_features, train_labels, test_features, test_labels, 
                          test_videos, models, feature_names):
    """Create and save a summary of results
    
    Args:
        train_features: Training feature matrix
        train_labels: Training labels
        test_features: Test feature matrix
        test_labels: Test labels
        test_videos: List of test video names
        models: Dictionary of models and predictions
        feature_names: List of feature names
    """
    # Count real and fake videos in the training and test sets
    train_fake_count = sum(1 for label in train_labels if label == 1)
    train_real_count = sum(1 for label in train_labels if label == 0)
    test_fake_count = sum(1 for label in test_labels if label == 1)
    test_real_count = sum(1 for label in test_labels if label == 0)
    
    with open(os.path.join(RESULTS_DIR, 'results_summary.txt'), 'w') as f:
        f.write("DEEPFAKE DETECTION USING FOURIER TRANSFORM ANALYSIS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training set: {len(train_features)} videos ({train_fake_count} fake, {train_real_count} real)\n")
        f.write(f"Test set: {len(test_features)} videos ({test_fake_count} fake, {test_real_count} real)\n")
        f.write(f"Total: {len(train_features) + len(test_features)} videos\n\n")
        
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

def visualize_fourier_examples(video_dirs, save_dir):
    """Visualize Fourier examples from the first frame of each video
    
    Args:
        video_dirs: List of video directories
        save_dir: Directory to save visualizations
    """
    if not video_dirs:
        return
    
    for video_dir in video_dirs[:2]:  # Only process first two videos
        video_name = os.path.basename(video_dir)
        
        # Find first frame
        frame_files = [f for f in os.listdir(video_dir) 
                      if f.startswith('frame_') and f.endswith(('.jpg', '.png'))]
        
        if not frame_files:
            continue
            
        frame_files.sort()
        frame_path = os.path.join(video_dir, frame_files[0])
        
        # Create visualization
        save_path = os.path.join(save_dir, f'fourier_example_{video_name}.png')
        visualize_fourier_example(frame_path, save_path)

def main():
    """Main function to run the Fourier deepfake detection pipeline"""
    print("=" * 80)
    print("DEEPFAKE DETECTION USING FOURIER TRANSFORM ANALYSIS")
    print("=" * 80)
    
    # Get video directories for training and testing
    print("\nSelecting video directories...")
    
    train_fake_dirs = get_video_dirs(TRAIN_FAKE_PATH, limit=TRAIN_FAKE_LIMIT)
    train_real_dirs = get_video_dirs(TRAIN_REAL_PATH, limit=TRAIN_REAL_LIMIT)
    test_fake_dirs = get_video_dirs(TEST_FAKE_PATH, limit=TEST_FAKE_LIMIT)
    test_real_dirs = get_video_dirs(TEST_REAL_PATH, limit=TEST_REAL_LIMIT)
    
    print(f"Training set: {len(train_fake_dirs)} fake, {len(train_real_dirs)} real videos")
    print(f"Test set: {len(test_fake_dirs)} fake, {len(test_real_dirs)} real videos")
    
    # Create visualizations of Fourier transforms
    print("\nGenerating Fourier transform visualizations...")
    visualize_fourier_examples(train_fake_dirs, RESULTS_DIR)
    visualize_fourier_examples(train_real_dirs, RESULTS_DIR)
    
    # Process videos to extract features
    print("\nExtracting features from training videos...")
    train_fake_features, train_fake_labels, train_fake_videos, feature_names = process_videos(
        train_fake_dirs, 'fake')
    
    train_real_features, train_real_labels, train_real_videos, _ = process_videos(
        train_real_dirs, 'real')
    
    # Process test videos
    print("\nExtracting features from test videos...")
    test_fake_features, test_fake_labels, test_fake_videos, _ = process_videos(
        test_fake_dirs, 'fake')
    
    test_real_features, test_real_labels, test_real_videos, _ = process_videos(
        test_real_dirs, 'real')
    
    # Combine training data
    if len(train_fake_features) > 0 and len(train_real_features) > 0:
        train_features = np.vstack((train_fake_features, train_real_features))
        train_labels = np.concatenate((train_fake_labels, train_real_labels))
        train_videos = train_fake_videos + train_real_videos
    elif len(train_fake_features) > 0:
        train_features = train_fake_features
        train_labels = train_fake_labels
        train_videos = train_fake_videos
    elif len(train_real_features) > 0:
        train_features = train_real_features
        train_labels = train_real_labels
        train_videos = train_real_videos
    else:
        print("Error: No valid training videos found. Exiting.")
        return
    
    # Combine test data
    if len(test_fake_features) > 0 and len(test_real_features) > 0:
        test_features = np.vstack((test_fake_features, test_real_features))
        test_labels = np.concatenate((test_fake_labels, test_real_labels))
        test_videos = test_fake_videos + test_real_videos
    elif len(test_fake_features) > 0:
        test_features = test_fake_features
        test_labels = test_fake_labels
        test_videos = test_fake_videos
    elif len(test_real_features) > 0:
        test_features = test_real_features
        test_labels = test_real_labels
        test_videos = test_real_videos
    else:
        print("Error: No valid test videos found. Exiting.")
        return
    
    # Combine all data for overall visualization
    all_features = np.vstack((train_features, test_features))
    all_labels = np.concatenate((train_labels, test_labels))
    all_videos = train_videos + test_videos
    
    # Print summary of extracted features
    train_fake_count = sum(1 for label in train_labels if label == 1)
    train_real_count = sum(1 for label in train_labels if label == 0)
    test_fake_count = sum(1 for label in test_labels if label == 1)
    test_real_count = sum(1 for label in test_labels if label == 0)
    
    print(f"\nFeatures extracted:")
    print(f"Training set: {len(train_features)} videos ({train_fake_count} fake, {train_real_count} real)")
    print(f"Test set: {len(test_features)} videos ({test_fake_count} fake, {test_real_count} real)")
    print(f"Total: {len(all_features)} videos")
    
    # Check if we have enough samples
    if train_fake_count < 3 or train_real_count < 3 or test_fake_count < 3 or test_real_count < 3:
        print("Warning: Not enough samples for each class. For best results, ensure at least 3 samples of each class in both training and test sets.")
    
    # Save the feature names for reference
    if feature_names:
        with open(os.path.join(RESULTS_DIR, 'feature_names.txt'), 'w') as f:
            for i, name in enumerate(feature_names):
                f.write(f"{i+1}. {name}\n")
    
    # Generate visualizations
    print("\nGenerating t-SNE visualizations...")
    visualize_tsne(train_features, train_labels, train_videos, 
                  os.path.join(RESULTS_DIR, 'train_tsne.png'))
    visualize_tsne(test_features, test_labels, test_videos, 
                  os.path.join(RESULTS_DIR, 'test_tsne.png'))
    visualize_tsne(all_features, all_labels, all_videos, 
                  os.path.join(RESULTS_DIR, 'all_tsne.png'))
    
    print("\nGenerating PCA visualizations...")
    visualize_pca(train_features, train_labels, train_videos, 
                 os.path.join(RESULTS_DIR, 'train_pca.png'), feature_names)
    visualize_pca(test_features, test_labels, test_videos, 
                 os.path.join(RESULTS_DIR, 'test_pca.png'), feature_names)
    visualize_pca(all_features, all_labels, all_videos, 
                 os.path.join(RESULTS_DIR, 'all_pca.png'), feature_names)
    
    # Train and evaluate models
    print("\nTraining and evaluating models...")
    models = train_and_evaluate_model(train_features, train_labels, test_features, test_labels, feature_names)
    
    # Analyze errors
    analyze_prediction_errors(test_features, test_labels, models['svm_predictions'], test_videos, "SVM")
    analyze_prediction_errors(test_features, test_labels, models['lr_predictions'], test_videos, "Logistic Regression")
    
    # Create summary of results
    create_results_summary(train_features, train_labels, test_features, test_labels, 
                          test_videos, models, feature_names)
    
    print(f"\nAnalysis complete! Results saved in the {RESULTS_DIR} directory.")

if __name__ == "__main__":
    main()