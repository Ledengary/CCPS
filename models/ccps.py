import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import random

# Define network architectures
class EmbeddingNet(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dims, activation='relu', dropout=0.1):
        super(EmbeddingNet, self).__init__()
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create layers list
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(self.activation)
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
        
        # Output embedding layer
        layers.append(nn.Linear(hidden_dims[-1], int(embed_dim)))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

# class EmbeddingNet(nn.Module):
#     def __init__(self, input_dim, embed_dim, hidden_dims):
#         super(EmbeddingNet, self).__init__()
#         hidden_dim = 64
#         print(f"EmbeddingNet - Input dim: {input_dim}, Embed dim: {embed_dim}, Hidden dims: {hidden_dims}")
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, embed_dim)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# Contrastive loss function
class MaxMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(MaxMarginLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        # Euclidean distance
        dist = torch.nn.functional.pairwise_distance(output1, output2)
        # Contrastive loss
        loss = torch.mean((1-label) * torch.pow(dist, 2) + 
                          (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss

# Dataset for contrastive learning
class MaxMarginDataset(Dataset):
    def __init__(self, features, labels, indices):
        self.features = torch.tensor(features[indices], dtype=torch.float32)
        self.labels = torch.tensor(labels[indices], dtype=torch.long)
        self.indices = indices
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        # Get anchor
        anchor_idx = idx
        anchor = self.features[anchor_idx]
        anchor_label = self.labels[anchor_idx]
        
        # Randomly select another sample
        other_idx = np.random.choice(len(self.indices))
        while other_idx == anchor_idx:
            other_idx = np.random.choice(len(self.indices))
        
        other = self.features[other_idx]
        other_label = self.labels[other_idx]
        
        # 1 if same label (negative pair), 0 if different label (positive pair)
        # This is opposite of traditional contrastive loss because we want to separate classes
        pair_label = torch.tensor(1 if anchor_label != other_label else 0, dtype=torch.float32)
        
        return anchor, other, pair_label
    
# Classifier model that uses the embedding model
class ClassifierWithEmbedding(nn.Module):
    def __init__(self, embedding_model, embed_dim, hidden_dims=None, num_classes=2, activation='relu', dropout=0.1, use_dropout=False):
        super(ClassifierWithEmbedding, self).__init__()
        
        self.embedding_model = embedding_model
        self.use_dropout = use_dropout
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create classifier layers
        classifier_layers = []
        
        if hidden_dims is not None and len(hidden_dims) > 0:
            # First layer from embeddings
            classifier_layers.append(nn.Linear(embed_dim, hidden_dims[0]))
            classifier_layers.append(self.activation)
            if use_dropout:
                classifier_layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                classifier_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                classifier_layers.append(self.activation)
                if use_dropout:
                    classifier_layers.append(nn.Dropout(dropout))
            
            # Output layer
            classifier_layers.append(nn.Linear(hidden_dims[-1], num_classes))
        else:
            # Simple linear classifier if no hidden dims
            classifier_layers.append(nn.Linear(embed_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x):
        # Get embeddings
        embeddings = self.embedding_model(x)
        
        # Apply classifier
        logits = self.classifier(embeddings)
        
        return logits, embeddings

# Dataset for classification
class ClassificationDataset(Dataset):
    def __init__(self, features, labels, indices):
        self.features = torch.tensor(features[indices], dtype=torch.float32)
        self.labels = torch.tensor(labels[indices], dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Classifier model that uses the embedding model
class ConvClassifierWithEmbedding(nn.Module):
    def __init__(self, embedding_model, embed_dim, hidden_dims=None, num_classes=2, activation='relu', dropout=0.2, use_dropout=True):
        super(ConvClassifierWithEmbedding, self).__init__()
        
        self.embedding_model = embedding_model
        self.use_dropout = use_dropout
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create classifier layers
        classifier_layers = []
        
        if hidden_dims is not None and len(hidden_dims) > 0:
            # First layer from embeddings
            classifier_layers.append(nn.Linear(embed_dim, hidden_dims[0]))
            classifier_layers.append(self.activation)
            if use_dropout:
                classifier_layers.append(nn.Dropout(dropout))
            
            # Hidden layers
            for i in range(len(hidden_dims) - 1):
                classifier_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                classifier_layers.append(self.activation)
                if use_dropout:
                    classifier_layers.append(nn.Dropout(dropout))
            
            # Output layer
            classifier_layers.append(nn.Linear(hidden_dims[-1], num_classes))
        else:
            # Simple linear classifier if no hidden dims
            classifier_layers.append(nn.Linear(embed_dim, num_classes))
        
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, x, seq_lengths=None):
        # Get embeddings
        embeddings = self.embedding_model(x, seq_lengths)
        
        # Apply classifier
        logits = self.classifier(embeddings)
        
        return logits, embeddings

# Dataset for classification with sequences
class SequenceClassificationDataset(Dataset):
    def __init__(self, samples, max_seq_length):
        self.samples = samples
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        features = sample['features_scaled']
        label = sample['label']
        seq_len = sample['seq_length']
        
        # Pad sequence to max length
        padded_features = np.zeros((self.max_seq_length, features.shape[1]))
        padded_features[:min(seq_len, self.max_seq_length)] = features[:min(seq_len, self.max_seq_length)]
        
        return (
            torch.tensor(padded_features, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(min(seq_len, self.max_seq_length), dtype=torch.long)
        )
    
class ConvEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dims, kernel_sizes, activation='relu', dropout=0.2):
        super(ConvEmbeddingNet, self).__init__()
        self.embed_dim = embed_dim
        
        # Define activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'mish':
            self.activation = nn.Mish()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        
        # First convolutional layer
        self.conv_layers.append(
            nn.Conv1d(input_dim, hidden_dims[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)
        )
        
        # Additional convolutional layers
        for i in range(1, len(hidden_dims)):
            self.conv_layers.append(
                nn.Conv1d(hidden_dims[i-1], hidden_dims[i], kernel_size=kernel_sizes[min(i, len(kernel_sizes)-1)], 
                          padding=kernel_sizes[min(i, len(kernel_sizes)-1)]//2)
            )
        
        # Pooling layer
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Final projection to embedding dimension
        self.fc = nn.Linear(hidden_dims[-1], embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, seq_lengths=None):
        # Input shape: [batch_size, seq_length, input_dim]
        # Transpose to [batch_size, input_dim, seq_length] for 1D convolution
        x = x.transpose(1, 2)
        
        # Apply convolutional layers
        for conv in self.conv_layers:
            x = self.activation(conv(x))
            x = self.dropout(x)
        
        # Apply pooling (handles variable sequence lengths)
        x = self.pool(x).squeeze(-1)
        
        # Project to embedding space
        x = self.fc(x)
        
        return x

# Define contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        
    def forward(self, output1, output2, label):
        # Euclidean distance
        dist = torch.nn.functional.pairwise_distance(output1, output2)
        # Contrastive loss
        loss = torch.mean((1-label) * torch.pow(dist, 2) + 
                         (label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2))
        return loss
    
    
# Create PyTorch datasets for sequence data
class SequenceContrastiveDataset(Dataset):
    def __init__(self, samples, max_seq_length):
        self.samples = samples
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get anchor sample
        anchor = self.samples[idx]
        anchor_features = anchor['features_scaled']
        anchor_label = anchor['label']
        anchor_seq_len = anchor['seq_length']
        
        # Randomly select another sample
        other_idx = np.random.choice(len(self.samples))
        while other_idx == idx:
            other_idx = np.random.choice(len(self.samples))
        
        other = self.samples[other_idx]
        other_features = other['features_scaled']
        other_label = other['label']
        other_seq_len = other['seq_length']
        
        # Pad sequences to max length
        anchor_padded = np.zeros((self.max_seq_length, anchor_features.shape[1]))
        other_padded = np.zeros((self.max_seq_length, other_features.shape[1]))
        
        # Fill with actual values
        anchor_padded[:min(anchor_seq_len, self.max_seq_length)] = anchor_features[:min(anchor_seq_len, self.max_seq_length)]
        other_padded[:min(other_seq_len, self.max_seq_length)] = other_features[:min(other_seq_len, self.max_seq_length)]
        
        # 1 if same label (negative pair), 0 if different label (positive pair)
        # This is opposite of traditional contrastive loss because we want to separate classes
        pair_label = 1 if anchor_label != other_label else 0
        
        return (
            torch.tensor(anchor_padded, dtype=torch.float32),
            torch.tensor(other_padded, dtype=torch.float32),
            torch.tensor(pair_label, dtype=torch.float32),
            torch.tensor(min(anchor_seq_len, self.max_seq_length), dtype=torch.long),
            torch.tensor(min(other_seq_len, self.max_seq_length), dtype=torch.long)
        )