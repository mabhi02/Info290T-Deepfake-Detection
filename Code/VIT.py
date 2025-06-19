import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Configuration parameters
CONFIG = {
    'num_frames': 16,            # Number of frames to sample from each video
    'image_size': 224,           # Size to resize frames to
    'patch_size': 16,            # Size of patches
    'num_classes': 2,            # Binary classification (fake or real)
    'dim': 128,                  # Embedding dimension (reduced from 1024)
    'depth': 6,                  # Number of transformer layers (reduced from 12)
    'heads': 8,                  # Number of attention heads (reduced from 16)
    'mlp_dim': 512,              # Dimensionality of the MLP layer (reduced from 3072)
    'dropout': 0.1,              # Dropout rate
    'emb_dropout': 0.1,          # Embedding dropout rate
    'batch_size': 4,             # Batch size
    'learning_rate': 3e-5,       # Learning rate
    'weight_decay': 0.01,        # Weight decay
    'num_epochs': 10,            # Number of training epochs
    'data_dir': r'C:\Users\athar\Documents\GitHub\testcv\finalData',     # Directory containing the data
    'max_train_samples': 100,    # Maximum number of training samples per class
    'max_test_samples': 25,      # Maximum number of test samples per class
}

# Frame extraction function
def extract_frames(video_path, num_frames):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate sampling indices to get evenly spaced frames
    if total_frames <= num_frames:
        # If video has fewer frames than required, duplicate frames
        indices = np.linspace(0, total_frames-1, num_frames, dtype=np.int32)
    else:
        # Otherwise, select evenly spaced frames
        indices = np.linspace(0, total_frames-1, num_frames, dtype=np.int32)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            # Convert from BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            # If reading frame fails, add a blank frame
            frames.append(np.zeros((CONFIG['image_size'], CONFIG['image_size'], 3), dtype=np.uint8))
    
    cap.release()
    
    # If we couldn't extract enough frames, duplicate the last one to reach the desired count
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else np.zeros((CONFIG['image_size'], CONFIG['image_size'], 3), dtype=np.uint8))
    
    return frames[:num_frames]  # Return exactly num_frames frames

# Video Dataset
class VideoDataset(Dataset):
    def __init__(self, data_dir, split, transform=None):
        """
        Args:
            data_dir (string): Directory with videos organized in subdirectories
            split (string): 'train' or 'test' directory
            transform (callable, optional): Transform to be applied on frames
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.videos = []
        self.labels = []
        
        # Map class names to integers
        self.class_to_idx = {'fake': 0, 'real': 1}
        
        # Load videos and labels
        split_dir = os.path.join(data_dir, split)
        
        # For limiting dataset size
        class_counters = {'fake': 0, 'real': 0}
        max_samples = CONFIG['max_train_samples'] if split == 'train' else CONFIG['max_test_samples']
        
        for class_name in os.listdir(split_dir):
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir) and class_name in self.class_to_idx:
                for filename in os.listdir(class_dir):
                    if self._is_video_file(filename):
                        # Skip if we've reached the maximum number of samples for this class
                        if class_counters[class_name] >= max_samples:
                            continue
                            
                        self.videos.append(os.path.join(class_dir, filename))
                        self.labels.append(self.class_to_idx[class_name])
                        class_counters[class_name] += 1
        
        # Shuffle the dataset
        indices = list(range(len(self.videos)))
        random.shuffle(indices)
        self.videos = [self.videos[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        
        print(f"Loaded {split} dataset with {class_counters['fake']} fake and {class_counters['real']} real videos")
    
    def _is_video_file(self, filename):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.MP4', '.AVI', '.MOV', '.MKV', '.WEBM']
        return any(filename.endswith(ext) for ext in video_extensions)
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        # Extract frames from the video
        try:
            frames = extract_frames(video_path, CONFIG['num_frames'])
            
            # Apply transformations to each frame
            if self.transform:
                frames = [self.transform(frame) for frame in frames]
                
            # Stack frames into a tensor
            frames = torch.stack(frames)
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            # Create a placeholder with zero frames in case of error
            frames = torch.zeros((CONFIG['num_frames'], 3, CONFIG['image_size'], CONFIG['image_size']))
        
        return frames, label

# Vision Transformer Model Components
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, dim):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape to process each frame
        x = x.reshape(batch_size * num_frames, channels, height, width)
        
        # Project into embedding space
        x = self.projection(x)  # (batch*frames, dim, h', w')
        
        # Flatten spatial dimensions
        x = x.flatten(2)  # (batch*frames, dim, h'*w')
        x = x.transpose(1, 2)  # (batch*frames, h'*w', dim)
        
        # Reshape back to include frames
        x = x.reshape(batch_size, num_frames, self.num_patches, -1)
        
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, dim)
        batch_size, seq_len, dim = x.shape
        
        # Get query, key, value projections
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = dots.softmax(dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class TemporalAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads, dropout)
        
    def forward(self, x):
        # x shape: (batch, frames, patches, dim)
        batch, frames, patches, dim = x.shape
        
        # Transpose to attend across frames
        x = x.transpose(1, 2)  # (batch, patches, frames, dim)
        x_flat = x.reshape(batch * patches, frames, dim)
        
        # Apply attention to frames
        out = x_flat + self.attn(self.norm(x_flat))
        
        # Reshape back
        out = out.reshape(batch, patches, frames, dim)
        out = out.transpose(1, 2)  # (batch, frames, patches, dim)
        
        return out

# Main Video Vision Transformer (ViT) Model
class VideoViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        image_size = config['image_size']
        patch_size = config['patch_size']
        num_frames = config['num_frames']
        num_classes = config['num_classes']
        dim = config['dim']
        depth = config['depth']
        heads = config['heads']
        mlp_dim = config['mlp_dim']
        dropout = config['dropout']
        emb_dropout = config['emb_dropout']
        
        # Calculate parameters
        num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(image_size, patch_size, 3, dim)
        
        # Position embeddings (for patches)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, num_patches + 1, dim))
        
        # Temporal position embeddings (for frames)
        self.temporal_embedding = nn.Parameter(torch.randn(1, num_frames, 1, dim))
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, dim))
        
        self.dropout = nn.Dropout(emb_dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([])
        for _ in range(depth):
            self.transformer_blocks.append(
                TransformerBlock(dim, heads, mlp_dim, dropout)
            )
        
        # Temporal attention blocks
        self.temporal_attention = nn.ModuleList([])
        for _ in range(depth // 2):  # Fewer temporal blocks
            self.temporal_attention.append(
                TemporalAttention(dim, heads, dropout)
            )
        
        # Layer Norm
        self.norm = nn.LayerNorm(dim)
        
        # Classifier head
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, frames, channels, height, width)
        batch_size, num_frames = x.shape[0], x.shape[1]
        
        # Get patch embeddings
        x = self.patch_embedding(x)  # (batch, frames, patches, dim)
        
        # Add CLS token to each frame
        cls_tokens = repeat(self.cls_token, '1 1 1 d -> b f 1 d', b=batch_size, f=num_frames)
        x = torch.cat((cls_tokens, x), dim=2)  # (batch, frames, patches+1, dim)
        
        # Add position embeddings
        x = x + self.pos_embedding
        
        # Add temporal embeddings
        x = x + self.temporal_embedding
        
        x = self.dropout(x)
        
        # Process each frame with transformer blocks
        for i, transformer in enumerate(self.transformer_blocks):
            # Flatten frames and patches for transformer
            b, f, p, d = x.shape
            x = x.reshape(b * f, p, d)  # Use reshape instead of view
            
            # Apply transformer block
            x = transformer(x)
            
            # Restore frame dimension
            x = x.reshape(b, f, p, d)  # Use reshape instead of view
            
            # Apply temporal attention after every other transformer block
            if i < len(self.temporal_attention) and i % 2 == 1:
                x = self.temporal_attention[i // 2](x)
        
        # Use the CLS tokens for classification
        x = x[:, :, 0]  # (batch, frames, dim)
        
        # Average across frames
        x = x.mean(dim=1)  # (batch, dim)
        
        x = self.to_latent(x)
        
        # Classification head
        return self.mlp_head(x)

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Add tqdm progress bar for training
        train_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, labels in train_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_loss = loss.item()
            current_acc = (predicted == labels).sum().item() / labels.size(0)
            train_progress.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_video_vit_model.pth')
            print('Model saved!')
    
    return train_losses, val_losses, train_accs, val_accs

# Evaluation function
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Add tqdm progress bar for evaluation
    eval_progress = tqdm(dataloader, desc='Evaluation')
    with torch.no_grad():
        for inputs, labels in eval_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_loss = loss.item()
            current_acc = (predicted == labels).sum().item() / labels.size(0)
            eval_progress.set_postfix({'loss': f'{current_loss:.4f}', 'acc': f'{current_acc:.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    
    return epoch_loss, epoch_acc

# Function to test the model and generate confusion matrix and classification report
def test_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    # Add tqdm progress bar for testing
    test_progress = tqdm(test_loader, desc='Testing')
    with torch.no_grad():
        for inputs, labels in test_progress:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with current accuracy
            if len(all_preds) > 0:
                current_acc = sum(np.array(all_preds) == np.array(all_labels)) / len(all_preds)
                test_progress.set_postfix({'acc': f'{current_acc:.4f}'})
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=['fake', 'real'])
    
    return cm, report

# Visualization functions
def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.close()

def plot_accuracies(train_accs, val_accs):
    plt.figure(figsize=(10, 5))
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.close()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['fake', 'real'], yticklabels=['fake', 'real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

# Main execution
# Save model info for later reference
def save_model_info(model, config):
    with open('model_info.txt', 'w') as f:
        f.write("=== Model Configuration ===\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n=== Model Architecture ===\n")
        f.write(str(model))
        
        f.write(f"\n\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
        
# Add this to the main function
def main():
    # Print the data directory being used
    print(f"Using data directory: {CONFIG['data_dir']}")
    print(f"Train directory: {os.path.join(CONFIG['data_dir'], 'train')}")
    print(f"Test directory: {os.path.join(CONFIG['data_dir'], 'test')}")
    print(f"Max training samples per class: {CONFIG['max_train_samples']}")
    print(f"Max test samples per class: {CONFIG['max_test_samples']}")
    
    # Verify directories exist
    train_dir = os.path.join(CONFIG['data_dir'], 'train')
    test_dir = os.path.join(CONFIG['data_dir'], 'test')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
    # Check for fake and real subdirectories
    for subdir in ['fake', 'real']:
        train_subdir = os.path.join(train_dir, subdir)
        test_subdir = os.path.join(test_dir, subdir)
        
        if not os.path.exists(train_subdir):
            raise FileNotFoundError(f"Train subdirectory not found: {train_subdir}")
        if not os.path.exists(test_subdir):
            raise FileNotFoundError(f"Test subdirectory not found: {test_subdir}")
    
    # Data transformations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = VideoDataset(CONFIG['data_dir'], 'train', transform=transform)
    test_dataset = VideoDataset(CONFIG['data_dir'], 'test', transform=transform)
    
    print(f"Number of training videos: {len(train_dataset)}")
    print(f"Number of test videos: {len(test_dataset)}")
    
    # Create data loaders (reduce num_workers if needed on your system)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize model
    print("Initializing model...")
    model = VideoViT(CONFIG).to(device)
    
    # Print and save model summary
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {num_params}")
    save_model_info(model, CONFIG)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    # Train model
    print("Starting training...")
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        CONFIG['num_epochs'], device
    )
    
    # Load best model and evaluate
    print("Loading best model for final evaluation...")
    model.load_state_dict(torch.load('best_video_vit_model.pth'))
    cm, report = test_model(model, test_loader, device)
    
    # Print final results
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Plot results
    print("Generating plots...")
    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accs, val_accs)
    plot_confusion_matrix(cm)
    
    print("Training and evaluation completed!")
    
    # Initialize model
    model = VideoViT(CONFIG).to(device)
    
    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    # Train model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, test_loader, criterion, optimizer, 
        CONFIG['num_epochs'], device
    )
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_video_vit_model.pth'))
    cm, report = test_model(model, test_loader, device)
    
    # Print final results
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    # Plot results
    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accs, val_accs)
    plot_confusion_matrix(cm)
    
    print("Training and evaluation completed!")

if __name__ == "__main__":
    main()