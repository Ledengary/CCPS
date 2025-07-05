import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from datetime import datetime
import json
import random

parser = argparse.ArgumentParser(description="Train a contrastive projection model for OE data")
parser.add_argument('--visible_cudas', type=str, default='0', help='Visible CUDA devices')
parser.add_argument('--feature_dir', type=str, required=True, help='Input data directory')
parser.add_argument('--train_dataset_name', type=str, required=True, help='Dataset name')
parser.add_argument('--test_dataset_name', type=str, required=True, help='Dataset name') # only used for proper location saving, no testing is done here
parser.add_argument('--llm_id', type=str, required=True, help='LLM ID')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
parser.add_argument('--max_seq_length', type=int, default=30, help='Maximum sequence length for tokens')
parser.add_argument('--hidden_dims', type=str, default='64,64', help='Hidden dimensions for convolutional layers (comma-separated)')
parser.add_argument('--embed_dim', type=int, default=4, help='Embedding dimension')
parser.add_argument('--kernel_sizes', type=str, default='3,3', help='Kernel sizes for convolutional layers (comma-separated)')
parser.add_argument('--activation', type=str, default='relu', help='Activation function')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
parser.add_argument('--loss_type', type=str, default='contrastive', help='Loss function type')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for contrastive loss')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--train_steps', type=int, default=5000, help='Total training steps')
parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation frequency in steps')
parser.add_argument('--log_steps', type=int, default=25, help='Logging frequency in steps')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay')
parser.add_argument('--seed', type=int, default=23, help='Random seed for reproducibility')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save trained model and outputs')
parser.add_argument('--visualize_3d', action='store_true', help='Generate 3D visualizations')
parser.add_argument('--umap_neighbors', type=int, default=15, help='UMAP n_neighbors parameter')
parser.add_argument('--umap_min_dist', type=float, default=0.1, help='UMAP min_dist parameter')
parser.add_argument('--eps_search_high', type=float, default=20, help='maximum eps searched for')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas

original_sys_path = sys.path.copy()
utils_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../utils"))
if utils_path not in sys.path:
    sys.path.append(utils_path)
from general import (
    set_visible_cudas, 
    seed_everything,
)
sys.path = original_sys_path
seed_everything(args.seed)
set_visible_cudas(args.visible_cudas)

from ccps import (
    ConvEmbeddingNet,
    ContrastiveLoss,
    SequenceContrastiveDataset,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def remove_eot_token_ids(data, llm_id):
    """Remove end-of-text tokens from the data"""
    def return_eot_token_ids(llm_id):
        if "llama" in llm_id.lower():
            return [128009]
        elif "mistral" in llm_id.lower():
            return [2]
        elif "qwen" in llm_id.lower():
            return [151645, 198]
        else:
            raise ValueError(f"Unknown LLM ID: {llm_id}")
    
    eot_token_ids = return_eot_token_ids(llm_id)
    data = data[~data['token_id'].isin(eot_token_ids)]
    return data

def prepare_sequential_data(df, max_seq_length, eps_search_high):
    """
    Group data by sample to create sequential representations.
    """
    # Columns to exclude from features
    exclude_cols = [
        'hash_id', 'task_name', 'sample_idx_in_task', 'token_idx_in_response',
        'token_str', 'token_id', 'query_label_sample', 'answer_type',
        'pei_curve_token', 'wrong_answer_idx',
    ]
    
    # Extract identifiers for each sample
    sample_identifiers = df[['hash_id', 'sample_idx_in_task', 'task_name', 'query_label_sample']].drop_duplicates()
    logger.info(f"Found {len(sample_identifiers)} unique samples")
    
    # Create a list to store processed samples
    processed_samples = []
    
    # Process each sample
    for idx, identifier in tqdm(sample_identifiers.iterrows(), total=len(sample_identifiers), desc="Processing samples"):
        sample_data = df[
            (df['hash_id'] == identifier['hash_id']) & 
            (df['sample_idx_in_task'] == identifier['sample_idx_in_task']) & 
            (df['task_name'] == identifier['task_name'])
        ]
        
        # Sort by token index to ensure correct sequence order
        sample_data = sample_data.sort_values('token_idx_in_response')
        
        # Extract features (drop exclude columns)
        feat_cols = [col for col in df.columns if col not in exclude_cols]
        features = sample_data[feat_cols].copy()
        
        # Replace inf values with eps_search_high for epsilon_to_flip_token
        if 'epsilon_to_flip_token' in features.columns:
            features['epsilon_to_flip_token'] = features['epsilon_to_flip_token'].replace([np.inf], eps_search_high)
        
        # Replace all inf/-inf with NaN
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Only include rows without NaN values
        mask = features.notna().all(axis=1)
        features = features[mask]
        
        # Skip sample if no valid tokens
        if len(features) == 0:
            continue
            
        # Extract label
        label = identifier['query_label_sample']
        
        # Save processed sample
        processed_samples.append({
            'hash_id': identifier['hash_id'],
            'sample_idx_in_task': identifier['sample_idx_in_task'],
            'task_name': identifier['task_name'],
            'features': features.values,  # Numpy array of features
            'seq_length': len(features),  # Actual sequence length before padding
            'label': label
        })
    
    return processed_samples

def load_and_preprocess_data(args):
    """Load and preprocess the dataset into sequences"""
    logger.info("Loading and preprocessing data...")
    
    # Construct feature path
    main_feature_path = os.path.join(args.feature_dir, args.train_dataset_name, args.llm_id.replace('/', '-'))
    
    # Get available tasks
    available_tasks = [
        name for name in os.listdir(main_feature_path)
        if os.path.isdir(os.path.join(main_feature_path, name))
    ]
    logger.info(f"Available tasks: {available_tasks}")
    
    # Load data from all tasks
    all_data = {}
    for task in available_tasks:
        task_path = os.path.join(main_feature_path, task, "llm_output_features.pkl")
        if os.path.exists(task_path):
            with open(task_path, "rb") as f:
                data = pickle.load(f)
        else:
            logger.warning(f"File path {task_path} does not exist. Skipping.")
            continue
        
        # Remove EOT tokens
        data = remove_eot_token_ids(data, args.llm_id)
        
        all_data[task] = data
        logger.info(f"Loaded {task} data with shape {data.shape}")
    
    logger.info(f"{len(all_data)} task data loaded successfully.")
    
    # Concatenate all tasks
    all_data_concat = pd.concat(all_data.values(), ignore_index=True)
    logger.info(f"Concatenated data shape: {all_data_concat.shape}")
    
    # Check for NaN values
    nan_counts = all_data_concat.isna().sum()
    logger.info(f"NaN counts per column (top 5): {nan_counts.sort_values(ascending=False).head()}")
    
    # Process data into sequences
    logger.info("Processing data into sequences...")
    sequences = prepare_sequential_data(all_data_concat, args.max_seq_length, args.eps_search_high)
    logger.info(f"Created {len(sequences)} sequences")
    
    # Split sequences into train/val 
    labels = np.array([seq['label'] for seq in sequences])
    train_indices, val_indices = train_test_split(
        np.arange(len(sequences)), test_size=args.val_ratio, 
        stratify=labels, random_state=args.seed
    )
    
    # Create a feature scaler based on training tokens
    train_sequences = [sequences[i] for i in train_indices]
    train_features = np.vstack([seq['features'] for seq in train_sequences])
    logger.info(f"Stacked TRAINING feature shape for scaling: {train_features.shape}")
    
    scaler = StandardScaler()
    scaler.fit(train_features)
    
    # Apply scaling to all samples
    for sample in sequences:
        sample['features_scaled'] = scaler.transform(sample['features'])
    
    train_data = [sequences[i] for i in train_indices]
    val_data = [sequences[i] for i in val_indices]
    
    logger.info(f"Data split - Train: {len(train_data)}, Validation: {len(val_data)}")
    
    # Print label distribution
    train_labels = np.array([s['label'] for s in train_data])
    val_labels = np.array([s['label'] for s in val_data])
    
    logger.info(f"Train label distribution: {np.bincount(train_labels.astype(int))}")
    logger.info(f"Validation label distribution: {np.bincount(val_labels.astype(int))}")
    
    # Get feature dimension
    feature_dim = sequences[0]['features'].shape[1]
    logger.info(f"Feature dimension: {feature_dim}")
    
    # Get feature column names from original dataframe
    exclude_cols = [
        'hash_id', 'task_name', 'sample_idx_in_task', 'token_idx_in_response',
        'token_str', 'token_id', 'query_label_sample', 'answer_type',
        'pei_curve_token', 'wrong_answer_idx',
    ]
    feature_columns = [col for col in all_data_concat.columns if col not in exclude_cols]
    
    return train_data, val_data, feature_dim, scaler, feature_columns

def create_dataloaders(train_data, val_data, args):
    """Create dataloaders for training"""
    if args.loss_type == 'contrastive':
        train_dataset = SequenceContrastiveDataset(train_data, args.max_seq_length)
        val_dataset = SequenceContrastiveDataset(val_data, args.max_seq_length)
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    logger.info(f"{args.loss_type} datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    logger.info(f"{args.loss_type} dataloaders - Train: {len(train_loader)}, Val: {len(val_loader)}")
    
    return train_loader, val_loader

def train_contrastive(model, train_loader, val_loader, optimizer, criterion, args):
    """Train the model using contrastive loss"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")
    model = model.to(device)
    best_val_loss = float('inf')
    best_model_state = None
    train_losses = []
    val_losses = []
    
    # Step counter
    step = 0
    max_steps = args.train_steps
    
    # Create progress bar
    pbar = tqdm(total=max_steps, desc="Training")
    
    # Training loop
    model.train()
    while step < max_steps:
        for anchor_seq, other_seq, label, anchor_len, other_len in train_loader:
            if step >= max_steps:
                break
            
            anchor_seq, other_seq, label = anchor_seq.to(device), other_seq.to(device), label.to(device)
            anchor_len, other_len = anchor_len.to(device), other_len.to(device)
            
            optimizer.zero_grad()
            anchor_out = model(anchor_seq, anchor_len)
            other_out = model(other_seq, other_len)
            
            loss = criterion(anchor_out, other_out, label)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Logging
            if step % args.log_steps == 0:
                avg_loss = np.mean(train_losses[-args.log_steps:]) if len(train_losses) >= args.log_steps else np.mean(train_losses)
                logger.info(f'Step {step}/{max_steps}, Train Loss: {avg_loss:.4f}')
            
            # Evaluation
            if step % args.eval_steps == 0:
                val_loss = evaluate_contrastive(model, val_loader, criterion, device)
                val_losses.append(val_loss)
                logger.info(f'Step {step}/{max_steps}, Validation Loss: {val_loss:.4f}')
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    logger.info(f'New best model at step {step} with validation loss: {val_loss:.4f}')
                
                # Switch back to training mode
                model.train()
            
            step += 1
            pbar.update(1)
    
    pbar.close()
    
    # Load the best model
    model.load_state_dict(best_model_state)
    logger.info(f'Training completed. Best validation loss: {best_val_loss:.4f}')
    
    return model, train_losses, val_losses, best_val_loss

def evaluate_contrastive(model, val_loader, criterion, device):
    """Evaluate model with contrastive loss"""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for anchor_seq, other_seq, label, anchor_len, other_len in val_loader:
            anchor_seq, other_seq, label = anchor_seq.to(device), other_seq.to(device), label.to(device)
            anchor_len, other_len = anchor_len.to(device), other_len.to(device)
            
            anchor_out = model(anchor_seq, anchor_len)
            other_out = model(other_seq, other_len)
            
            loss = criterion(anchor_out, other_out, label)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def generate_embeddings(model, data, device, batch_size=32, max_seq_length=30):
    """Generate embeddings for all samples in data"""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch_samples = data[i:i+batch_size]
        batch_features = [s['features_scaled'] for s in batch_samples]
        batch_labels = [s['label'] for s in batch_samples]
        batch_seq_lens = [min(s['seq_length'], max_seq_length) for s in batch_samples]
        
        # Get feature dimension
        feature_dim = batch_features[0].shape[1] if len(batch_features) > 0 else 0
        
        # Pad sequences
        batch_padded = np.zeros((len(batch_features), max_seq_length, feature_dim))
        for j, (features, seq_len) in enumerate(zip(batch_features, batch_seq_lens)):
            batch_padded[j, :seq_len, :] = features[:seq_len]
        
        # Convert to tensors
        batch_tensor = torch.tensor(batch_padded, dtype=torch.float32).to(device)
        seq_lens_tensor = torch.tensor(batch_seq_lens, dtype=torch.long).to(device)
        
        # Get embeddings
        with torch.no_grad():
            embeddings = model(batch_tensor, seq_lens_tensor)
        
        # Store results
        all_embeddings.append(embeddings.cpu().numpy())
        all_labels.extend(batch_labels)
    
    # Concatenate results
    embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    labels = np.array(all_labels)
    
    return embeddings, labels

def visualize_embeddings(model, data, output_dir, args, name=""):
    """Create UMAP visualizations of the embeddings"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {name} data...")
    embeddings, labels = generate_embeddings(model, data, device, args.batch_size, args.max_seq_length)
    
    # 2D UMAP
    logger.info(f"Creating 2D UMAP visualization for {name} data...")
    reducer_2d = umap.UMAP(n_components=2, random_state=args.seed, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
    embedding_2d = reducer_2d.fit_transform(embeddings)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.scatter(embedding_2d[labels==1, 0], embedding_2d[labels==1, 1], c='blue', label='Label 1', alpha=0.5)
    plt.scatter(embedding_2d[labels==0, 0], embedding_2d[labels==0, 1], c='red', label='Label 0', alpha=0.5)
    plt.title(f"{name} 2D UMAP Visualization")
    plt.legend()
    
    # Save figure
    output_file = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_umap_2d.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3D UMAP if requested
    if args.visualize_3d:
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            
            logger.info(f"Creating 3D UMAP visualization for {name} data...")
            reducer_3d = umap.UMAP(n_components=3, random_state=args.seed, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)
            embedding_3d = reducer_3d.fit_transform(embeddings)
            
            # Create 3D visualization
            fig = go.Figure()
            
            # Positive class
            fig.add_trace(
                go.Scatter3d(
                    x=embedding_3d[labels==1, 0],
                    y=embedding_3d[labels==1, 1],
                    z=embedding_3d[labels==1, 2],
                    mode='markers',
                    marker=dict(color='blue', size=3),
                    name='Label 1'
                )
            )
            
            # Negative class
            fig.add_trace(
                go.Scatter3d(
                    x=embedding_3d[labels==0, 0],
                    y=embedding_3d[labels==0, 1],
                    z=embedding_3d[labels==0, 2],
                    mode='markers',
                    marker=dict(color='red', size=3),
                    name='Label 0'
                )
            )
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='UMAP1',
                    yaxis_title='UMAP2',
                    zaxis_title='UMAP3'
                ),
                title=f"{name} 3D UMAP Visualization",
                width=800,
                height=600
            )
            
            # Save as HTML
            output_file = os.path.join(output_dir, f"{name.lower().replace(' ', '_')}_umap_3d.html")
            plot(fig, filename=output_file, auto_open=False)
        except ImportError:
            logger.warning("Plotly not installed. Skipping 3D visualization.")
    
    return embedding_2d

def plot_loss_curves(train_losses, val_losses, eval_steps, log_steps, output_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(np.arange(len(train_losses)) * log_steps, train_losses, label='Training Loss')
    
    # Plot validation loss
    eval_steps_x = np.arange(len(val_losses)) * eval_steps
    plt.plot(eval_steps_x, val_losses, 'o-', label='Validation Loss')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    output_file = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curves saved to {output_file}")

def save_model_and_config(model, scaler, feature_columns, args, stats, output_dir):
    """Save model, scaler, and configuration to output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "contrastive_oe_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved to {scaler_path}")
    
    # Save feature columns
    columns_path = os.path.join(output_dir, "feature_columns.json")
    with open(columns_path, "w") as f:
        json.dump(feature_columns, f)
    logger.info(f"Feature columns saved to {columns_path}")
    
    # Save configuration
    config = vars(args)
    
    # Add model architecture details
    config["model"] = {
        "input_dim": model.conv_layers[0].in_channels,
        "hidden_dims": [layer.out_channels for layer in model.conv_layers],
        "embed_dim": model.embed_dim,
        "kernel_sizes": [layer.kernel_size[0] for layer in model.conv_layers],
        "dropout": model.dropout.p
    }
    
    # Add training statistics
    config["stats"] = stats
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")
    logger.info(f"Config saved: {config}")

def setup_path():
    output_dir = os.path.join(args.output_dir, f"{args.train_dataset_name}_{args.test_dataset_name}", args.llm_id.replace('/', '-'))
    return output_dir

def main():
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = setup_path()
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup file logging
    file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Log configuration
    logger.info(f"Starting training with configuration: {args}")
    
    # Load and preprocess data
    train_data, val_data, feature_dim, scaler, feature_columns = load_and_preprocess_data(args)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(train_data, val_data, args)
    
    # Parse architecture parameters
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    
    # Create model
    model = ConvEmbeddingNet(
        input_dim=feature_dim, 
        embed_dim=args.embed_dim,
        hidden_dims=hidden_dims,
        kernel_sizes=kernel_sizes,
        activation=args.activation,
        dropout=args.dropout
    )
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create loss function and train model
    if args.loss_type == 'contrastive':
        criterion = ContrastiveLoss(margin=args.margin)
        model, train_losses, val_losses, best_val_loss = train_contrastive(
            model, train_loader, val_loader, optimizer, criterion, args
        )
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, args.eval_steps, args.log_steps, output_dir)
    
    # Visualize embeddings
    logger.info("Generating embedding visualizations...")
    train_vis = visualize_embeddings(model, train_data, output_dir, args, name="Training")
    val_vis = visualize_embeddings(model, val_data, output_dir, args, name="Validation")
    
    # Save model and configuration
    stats = {
        "best_val_loss": best_val_loss,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "train_sequence_stats": {
            "min_length": min(s['seq_length'] for s in train_data),
            "max_length": max(s['seq_length'] for s in train_data),
            "mean_length": sum(s['seq_length'] for s in train_data) / len(train_data)
        },
        "val_sequence_stats": {
            "min_length": min(s['seq_length'] for s in val_data),
            "max_length": max(s['seq_length'] for s in val_data),
            "mean_length": sum(s['seq_length'] for s in val_data) / len(val_data)
        },
        "training_steps": len(train_losses) * args.log_steps
    }
    
    save_model_and_config(model, scaler, feature_columns, args, stats, output_dir)
    
    logger.info("Training completed successfully!")
    return model, scaler, feature_columns

if __name__ == "__main__":
    main()


## Large llm_ids:
# unsloth/Meta-Llama-3.1-8B-Instruct
# unsloth/Qwen2.5-14B-Instruct
# unsloth/Mistral-Small-24B-Instruct-2501
# unsloth/Qwen2.5-32B-Instruct

# python ccps_oe_contrastive_train.py \
#   --visible_cudas 0 \
#   --feature_dir "../features" \
#   --train_dataset_name "CT-OE-WRG" \
#   --test_dataset_name "MMLU-OE" \
#   --llm_id "unsloth/Qwen2.5-32B-Instruct" \
#   --val_ratio 0.1 \
#   --hidden_dims "64,32,16" \
#   --kernel_sizes "3,3,3" \
#   --embed_dim 16 \
#   --activation "relu" \
#   --dropout 0.05 \
#   --loss_type contrastive \
#   --margin 1.0 \
#   --batch_size 32 \
#   --train_steps 5000 \
#   --eval_steps 500 \
#   --log_steps 25 \
#   --lr 1e-3 \
#   --weight_decay 0.01 \
#   --seed 23 \
#   --num_workers 4 \
#   --max_seq_length 30 \
#   --output_dir "../trained_models/CCPS/contrastive_ov" \
#   --umap_neighbors 15 \
#   --umap_min_dist 0.1
