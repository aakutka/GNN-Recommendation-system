import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import LGConv
from torch_geometric.utils import to_undirected
import numpy as np
from collections import defaultdict
import time
import logging
graph = torch.load('movielens_graph.pt')
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import LGConv

from torch_geometric.utils import to_undirected

import numpy as np

from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def forward(self, edge_index):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        layer_embeddings = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            layer_embeddings.append(x)
        
        final_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)

        user_embeddings, item_embeddings = torch.split(final_embeddings, [self.num_users, self.num_items])
        return user_embeddings, item_embeddings

    def predict(self, users, items):
        user_embeddings, item_embeddings = self(self.edge_index)
        user_emb = user_embeddings[users]
        item_emb = item_embeddings[items]
        return (user_emb * item_emb).sum(dim=1)








"""

import torch
import torch.nn as nn
from torch_geometric.nn import LGConv
from torch_sparse import SparseTensor

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, num_layers):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

        self.convs = nn.ModuleList([LGConv() for _ in range(num_layers)])

    def forward(self, edge_index, batch_size=10000):
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        edge_index = edge_index.to(x.device)
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(x.size(0), x.size(0)))

        layer_embeddings = [x]
        for conv in self.convs:
            xs = []
            for batch_start in range(0, x.size(0), batch_size):
                batch_end = min(batch_start + batch_size, x.size(0))
                x_batch = x[batch_start:batch_end]
                x_batch = conv(x_batch, adj[batch_start:batch_end])
                xs.append(x_batch)
            x = torch.cat(xs, dim=0)
            layer_embeddings.append(x)
        
        final_embeddings = torch.stack(layer_embeddings, dim=1).mean(dim=1)

        user_embeddings, item_embeddings = torch.split(final_embeddings, [self.num_users, self.num_items])
        return user_embeddings, item_embeddings

    def predict(self, users, items):
        user_embeddings, item_embeddings = self(self.edge_index)
        user_emb = user_embeddings[users]
        item_emb = item_embeddings[items]
        return (user_emb * item_emb).sum(dim=1)

def train_lightgcn(model, edge_index, optimizer, num_epochs, batch_size):
    model.train()
    
    num_users = model.num_users
    num_items = model.num_items
    
    for epoch in range(num_epochs):
        total_loss = 0
        for _ in range(edge_index.size(1) // batch_size + 1):
            optimizer.zero_grad()
            
            # Sample edges
            perm = torch.randperm(edge_index.size(1))[:batch_size]
            edge_index_batch = edge_index[:, perm]
            
            users = edge_index_batch[0]
            pos_items = edge_index_batch[1]
            neg_items = torch.randint(0, num_items, (batch_size,), device=edge_index.device)
            
            user_emb, item_emb = model(edge_index)
            
            user_emb = user_emb[users]
            pos_item_emb = item_emb[pos_items]
            neg_item_emb = item_emb[neg_items]
            
            pos_scores = (user_emb * pos_item_emb).sum(dim=1)
            neg_scores = (user_emb * neg_item_emb).sum(dim=1)
            
            loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# Main execution



















def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_scores = (user_emb * pos_item_emb).sum(dim=1)
    neg_scores = (user_emb * neg_item_emb).sum(dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
    return loss


def train_and_evaluate_lightgcn(model, edge_index, train_data, test_data, optimizer, num_epochs, batch_size, k=10):
    model.train()
    
    num_users = model.num_users
    num_items = model.num_items
    
    for epoch in range(num_epochs):
        total_loss = 0
        for _ in range(len(train_data) // batch_size + 1):
            optimizer.zero_grad()
            
            batch = train_data[torch.randint(0, len(train_data), (batch_size,))]
            users, pos_items = batch[:, 0], batch[:, 1]
            neg_items = torch.randint(0, num_items, (batch_size,))
            
            user_emb, item_emb = model(edge_index)
            
            user_emb = user_emb[users]
            pos_item_emb = item_emb[pos_items]
            neg_item_emb = item_emb[neg_items]
            
            loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")
        
        if (epoch + 1) % 5 == 0:  # Evaluate every 5 epochs
            metrics = evaluate_model(model, test_data, k)
            print(f"Evaluation metrics: {metrics}")

    return model
def evaluate_model(model, test_data, k=10):
    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(model.edge_index)
    
    ndcgs, recalls, precisions = [], [], []
    
    for user, true_items in test_data.items():
        user_tensor = torch.tensor([user]).repeat(model.num_items)
        items_tensor = torch.arange(model.num_items)
        
        scores = model.predict(user_tensor, items_tensor)
        _, indices = torch.topk(scores, k)
        recommended_items = indices.tolist()
        
        ndcgs.append(ndcg_at_k(true_items, recommended_items, k))
        recalls.append(recall_at_k(true_items, recommended_items, k))
        precisions.append(precision_at_k(true_items, recommended_items, k))
    
    return {
        f'NDCG@{k}': np.mean(ndcgs),
        f'Recall@{k}': np.mean(recalls),
        f'Precision@{k}': np.mean(precisions)
    }



# Evaluation metric functions (as provided earlier)
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))

def ndcg_at_k(actual, predicted, k):
    if not actual:
        return 0.0
    return dcg_at_k([1 if i in actual else 0 for i in predicted], k) / dcg_at_k([1] * len(actual), k)

def recall_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(act_set))
    return result

def precision_at_k(actual, predicted, k):
    pred_set = set(predicted[:k])
    act_set = set(actual)
    result = len(act_set & pred_set) / float(k)
    return result


from collections import defaultdict

import torch
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from collections import defaultdict

def memory_efficient_to_undirected(edge_index, num_nodes=None):
    row, col = edge_index
    row, col = torch.cat([row, col]), torch.cat([col, row])
    edge_index = torch.stack([row, col], dim=0)
    return edge_index.to(torch.long)


"""

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the saved graph
    print("Loading graph...")
    graph = torch.load('movielens_graph.pt', map_location='cpu')  # Load to CPU
    print("Graph loaded successfully.")

    # Prepare edge_index for user-item interactions
    edge_index = graph['user', 'rates', 'movie'].edge_index

    # Make the graph undirected (as per LightGCN paper) on CPU
    print("Making graph undirected...")
    edge_index = memory_efficient_to_undirected(edge_index)
    print("Graph made undirected.")

    num_users = graph['user'].num_nodes
    num_items = graph['movie'].num_nodes
    print(f"Number of users: {num_users}, Number of items: {num_items}")

    # Prepare training data
    print("Preparing training data...")
    all_interactions = edge_index.t().numpy()
    train_data, test_data = train_test_split(all_interactions, test_size=0.2, random_state=42)
    train_data = torch.LongTensor(train_data)  # Keep on CPU for now

    # Prepare test data
    test_data_dict = defaultdict(list)
    for user, item in test_data:
        test_data_dict[user].append(item)
    test_data = dict(test_data_dict)
    print("Data preparation complete.")

    # Initialize LightGCN model
    model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3)
    model.edge_index = edge_index  # Store edge_index in the model for easy access

    # Move model and data to GPU if available
    if torch.cuda.is_available():
        print("Moving model and data to GPU...")
        model = model.to(device)
        edge_index = edge_index.to(device)
        train_data = train_data.to(device)
        print("Model and data moved to GPU.")
    else:
        print("GPU not available. Using CPU.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate the model
    print("Starting training and evaluation...")
    trained_model = train_and_evaluate_lightgcn(model, edge_index, train_data, test_data, optimizer, num_epochs=1, batch_size=1024, k=10)

    # Final evaluation
    print("Performing final evaluation...")
    final_metrics = evaluate_model(trained_model, test_data, k=10)
    print("Final Evaluation Metrics:", final_metrics)
"""






# Main execution
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading graph...")
    graph = torch.load('movielens_graph.pt', map_location='cpu')
    print("Graph loaded successfully.")

    edge_index = graph['user', 'rates', 'movie'].edge_index
    num_users = graph['user'].num_nodes
    num_items = graph['movie'].num_nodes

    print(f"Number of users: {num_users}, Number of items: {num_items}")

    # Initialize model
    model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=3)
    model.edge_index = edge_index.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    print("Starting training...")
    train_lightgcn(model, edge_index, optimizer, num_epochs=50, batch_size=1024*64)

    print("Training completed.")






























