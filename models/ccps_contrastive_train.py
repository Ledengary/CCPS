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
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
from datetime import datetime
import json

parser = argparse.ArgumentParser(description="Train a contrastive projection model on MC data")
parser.add_argument('--visible_cudas', type=str, default='0', help='Visible CUDA devices')
parser.add_argument('--feature_dir', type=str, required=True, help='Input data directory')
parser.add_argument('--train_dataset_name', type=str, required=True, help='Dataset name')
parser.add_argument('--test_dataset_name', type=str, required=True, help='Dataset name') # only used for proper location saving, no testing is done here
parser.add_argument('--llm_id', type=str, required=True, help='LLM ID')
parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
parser.add_argument('--hidden_dims', type=str, default='64,64', help='Hidden layer dimensions (comma-separated)')
parser.add_argument('--embed_dim', type=str, default=4, help='Embedding dimension')
parser.add_argument('--activation', type=str, default='relu', help='Activation function')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--loss_type', type=str, default='contrastive', help='Loss function type')
parser.add_argument('--margin', type=float, default=1.0, help='Margin for contrastive loss')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
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
    EmbeddingNet,
    MaxMarginLoss,
    MaxMarginDataset,
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

def load_and_preprocess_data(args):
    """Load and preprocess the dataset"""
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
    
    # Extract features and labels
    exclude_cols = [
        'hash_id', 'task_name', 'sample_idx_in_task', 'token_idx_in_response',
        'token_str', 'token_id', 'query_label_sample', 'answer_type',
        'pei_curve_token', 'wrong_answer_idx',
    ]
    
    # Extract features
    feat_df = all_data_concat.drop(columns=exclude_cols)
    # now specifically for the epsilon_to_flip_token column, if inf, replace with args.eps_search_high
    feat_df['epsilon_to_flip_token'] = feat_df['epsilon_to_flip_token'].replace([np.inf], args.eps_search_high)
    # now count the number of -inf and inf values in each column
    inf_counts = (feat_df == np.inf).sum()
    logger.info(f"Inf counts per column (before replacement):\n{inf_counts.sort_values(ascending=False).head()}")
    ninf_counts = (feat_df == -np.inf).sum()
    logger.info(f"-Inf counts per column (before replacement):\n{ninf_counts.sort_values(ascending=False).head()}")
    feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # now tell me how many NaN values are in each column
    logger.info(f"NaN counts per column (after replacement):\n{feat_df.isna().sum().sort_values(ascending=False).head()}")
    mask = feat_df.notna().all(axis=1)
    feat_df = feat_df[mask]
    
    # Get labels separately (using the same mask)
    labels_clean = all_data_concat.loc[mask, 'query_label_sample'].values
    
    # Split data into train/val
    idx_all = np.arange(len(feat_df))
    idx_train, idx_val = train_test_split(
        idx_all, test_size=args.val_ratio, stratify=labels_clean, random_state=args.seed
    )
    
    # Create scaler and fit on training data
    scaler = StandardScaler()
    scaler.fit(feat_df.iloc[idx_train])
    
    # Transform both train and validation data using the training statistics
    features_scaled = scaler.transform(feat_df)
    
    logger.info(f"Data split - Train: {len(idx_train)}, Validation: {len(idx_val)}")
    
    # Save feature column names and scaler for later use
    feature_columns = feat_df.columns.tolist()
    
    # Print label distribution
    label_counts = np.bincount(labels_clean.astype(int))
    logger.info(f"Label distribution: {label_counts}")
    
    # Print split info
    logger.info(f"Training features shape: {features_scaled[idx_train].shape}")
    logger.info(f"Validation features shape: {features_scaled[idx_val].shape}")
    
    return features_scaled, labels_clean, idx_train, idx_val, scaler, feature_columns

def create_dataloaders(features, labels, train_idx, val_idx, args):
    """Create dataloaders for training"""
    if args.loss_type == 'contrastive':
        train_dataset = MaxMarginDataset(features, labels, train_idx)
        val_dataset = MaxMarginDataset(features, labels, val_idx)
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
        for anchor, other, label in train_loader:
            if step >= max_steps:
                break
            
            anchor, other, label = anchor.to(device), other.to(device), label.to(device)
            
            optimizer.zero_grad()
            anchor_out = model(anchor)
            other_out = model(other)
            
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
        for anchor, other, label in val_loader:
            anchor, other, label = anchor.to(device), other.to(device), label.to(device)
            
            anchor_out = model(anchor)
            other_out = model(other)
            
            loss = criterion(anchor_out, other_out, label)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)


def visualize_embeddings(model, features, labels, output_dir, name=""):
    """Create UMAP visualizations of the embeddings"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Generate embeddings
    with torch.no_grad():
        embeddings = model(torch.tensor(features, dtype=torch.float32).to(device)).cpu().numpy()
    
    # 2D UMAP
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

def save_model_and_config(model, scaler, feature_columns, args, stats, output_dir):
    """Save model, scaler, and configuration to output directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "contrastive_model.pt")
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
        "input_dim": model.model[0].in_features,
        "hidden_dims": [model.model[i].out_features for i in range(0, len(model.model), 3) if isinstance(model.model[i], nn.Linear)][:-1],
        "embed_dim": model.model[-1].out_features,
        "activation": str(model.activation.__class__.__name__),
        "dropout": config["dropout"]
    }
    
    # Add training statistics
    config["stats"] = stats
    
    # Save config
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved to {config_path}")

def plot_loss_curves(train_losses, val_losses, eval_steps, output_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    
    # Plot training loss
    plt.plot(np.arange(len(train_losses)) * args.log_steps, train_losses, label='Training Loss')
    
    # Plot validation loss
    eval_steps_x = np.arange(len(val_losses)) * args.eval_steps
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
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    # Load and preprocess data
    features, labels, train_idx, val_idx, scaler, feature_columns = load_and_preprocess_data(args)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(features, labels, train_idx, val_idx, args)
    
    # Create model
    input_dim = features.shape[1]
    model = EmbeddingNet(input_dim, args.embed_dim, hidden_dims, args.activation, args.dropout)
    # model = EmbeddingNet(input_dim, int(args.embed_dim), hidden_dims)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create loss function
    if args.loss_type == 'contrastive':
        criterion = MaxMarginLoss(margin=args.margin)
        model, train_losses, val_losses, best_val_loss = train_contrastive(model, train_loader, val_loader, optimizer, criterion, args)
    else:
        raise ValueError(f"Unsupported loss type: {args.loss_type}")
    
    # Plot loss curves
    plot_loss_curves(train_losses, val_losses, args.eval_steps, output_dir)

    # Visualize embeddings
    logger.info("Generating embedding visualizations...")
    # train_vis = visualize_embeddings(model, features[train_idx], labels[train_idx], output_dir, name="Training")
    # val_vis = visualize_embeddings(model, features[val_idx], labels[val_idx], output_dir, name="Validation")
    
    # Save model and configuration
    stats = {
        "best_val_loss": best_val_loss,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
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

# python ccps_contrastive_train.py \
#   --visible_cudas 0 \
#   --feature_dir "../features" \
#   --train_dataset_name "CT-CHOICE-WRG" \
#   --test_dataset_name "MMLU-PRO-CHOICE" \
#   --llm_id "unsloth/Meta-Llama-3.1-8B-Instruct" \
#   --val_ratio 0.1 \
#   --hidden_dims "64,32,16" \
#   --embed_dim "8" \
#   --activation "elu" \
#   --dropout "0.05" \
#   --loss_type contrastive \
#   --margin 1.0 \
#   --batch_size 64 \
#   --train_steps 5000 \
#   --eval_steps 500 \
#   --log_steps 25 \
#   --lr 1e-4 \
#   --weight_decay 0.01 \
#   --seed 23 \
#   --num_workers 1 \
#   --output_dir "../trained_models/CCPS/contrastive_ov" \
#   --umap_neighbors 15 \
#   --umap_min_dist 0.1
