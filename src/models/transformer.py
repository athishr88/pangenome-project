import torch.nn.functional as F
from torch import nn
import torch
import os

device = 'cuda'

class Head(nn.Module):
    """One head of self attention"""

    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size)
        self.query = nn.Linear(n_embd, head_size)
        self.value = nn.Linear(n_embd, head_size)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)

        # Compute attention weights
        wei = q @ k.transpose(-2, -1)
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        out = wei @ v
        return out

    def get_weights(self, x):
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1)
        wei = F.softmax(wei, dim=-1)
        return wei

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, nm_heads,
                 head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(
            n_embd, head_size) for _ in range(nm_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    def get_weights_stacked(self, x):
        weights = [h.get_weights(x) for h in self.heads]
        return torch.stack(weights, dim=0)

class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd,
                                     n_head, head_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

    def get_attention_matrix(self, x):
        return self.sa.get_weights_stacked(self.ln1(x))

class BlocksSequential(nn.Module):
    def __init__(self, n_embd, n_head, dropout, n_layer):
        super().__init__()
        self.first_block = Block(n_embd, n_head, dropout)
        self.rest_blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer - 1)])

    def forward(self, x):
        x = self.first_block(x)
        x = self.rest_blocks(x)
        return x

    def get_attention_matrix(self, x):
        return self.first_block.get_attention_matrix(x)

class PangenomeTransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vocab_size = cfg.model.model_params.vocab_size
        n_embd = cfg.model.model_params.n_embd
        n_head = cfg.model.model_params.n_head
        dropout = cfg.model.model_params.dropout
        n_layer = cfg.model.model_params.n_layer
        hidden_size = cfg.model.model_params.hidden_size
        input_size = cfg.preprocessing.dataset.input_size
        num_classes = cfg.preprocessing.dataset.num_classes

        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.blocks = BlocksSequential(n_embd, n_head, dropout, n_layer)
        self.linear_comb = nn.Linear(n_embd, 1)  # TODO 1 can be changed to any number
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(input_size, hidden_size) #TODO 1788 can be changed to any number, define in the hyperparameters
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, X):
        out = F.relu(self.token_embeddings_table(X))  # (batch_size, seq_len, n_embd)
        out = self.blocks(out)  # (batch_size, seq_len, n_embd)
        
        out = self.linear_comb(out)  # (batch_size, seq_len, 1)
        out = out.squeeze(2)  # Convert to (batch_size, seq_len)
        # out = self.ln1(out)  # (batch_size, seq_len)
        
        x = self.dropout(F.relu(self.linear1(out)))
        x = self.linear2(x)
        return x

    def get_attention_matrix(self, X):
        X = F.relu(self.token_embeddings_table(X))
        out = self.blocks.get_attention_matrix(X)
        return out
    
class PGTransformerV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        vocab_size_sparse = cfg.model.model_params.vocab_size_sparse
        n_embd = cfg.model.model_params.n_embd
        self.n_embd = n_embd
        n_head = cfg.model.model_params.n_head
        dropout = cfg.model.model_params.dropout
        n_layer = cfg.model.model_params.n_layer
        hidden_size = cfg.model.model_params.hidden_size
        input_size = cfg.preprocessing.dataset.input_size
        num_classes = cfg.preprocessing.dataset.num_classes

        self.sparse_embeddings_table = nn.Embedding(vocab_size_sparse, n_embd)
        self.indices_embeddings_table = nn.Embedding(input_size, n_embd)
        self.blocks = BlocksSequential(n_embd, n_head, dropout, n_layer)
        self.linear_comb = nn.Linear(n_embd, 1)  # TODO 1 can be changed to any number
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.linear1 = nn.Linear(input_size, hidden_size) #TODO 1788 can be changed to any number, define in the hyperparameters
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, f, n, x):
        sparse_embd = F.relu(self.sparse_embeddings_table(x))  # (batch_size, seq_len, n_embd)
        indices_embd = F.relu(self.indices_embeddings_table(f))  # (batch_size, seq_len, n_embd)
        n = n.unsqueeze(-1).expand(-1, -1, self.n_embd)
        out = sparse_embd + indices_embd + n  # (batch_size, seq_len, n_embd) 
        out = self.blocks(out)  # (batch_size, seq_len, n_embd)
        
        out = self.linear_comb(out)  # (batch_size, seq_len, 1)
        out = out.squeeze(2)  # Convert to (batch_size, seq_len)
        # out = self.ln1(out)  # (batch_size, seq_len)
        
        x = self.dropout(F.relu(self.linear1(out)))
        x = self.linear2(x)
        return x


class PangenomeWindowedTransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = cfg.model.model_params.vocab_size
        n_embd = cfg.model.model_params.n_embd
        n_head = cfg.model.model_params.n_head
        dropout = cfg.model.model_params.dropout
        n_layer = cfg.model.model_params.n_layer
        hidden_size = cfg.model.model_params.hidden_size
        num_classes = cfg.preprocessing.dataset.num_classes
        block_size = cfg.preprocessing.dataset.window_size

        self.token_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = BlocksSequential(n_embd, n_head, dropout, n_layer)
        self.linear_comb = nn.Linear(n_embd, 1)  # TODO 1 can be changed to any number
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(block_size)
        self.linear1 = nn.Linear(block_size, hidden_size) #TODO 1788 can be changed to any number, define in the hyperparameters
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, indices, sparse_vals): #(5, 34), (5, 34)
        normalizer_val = self.cfg.preprocessing.dataset.normalizer_val
        indices = indices / normalizer_val # 236000
        tok_emb = self.token_embeddings_table(sparse_vals) # (5, 34, 16)

        B, T, C = tok_emb.shape
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (67, 16)
        pos_emb_expanded = pos_emb.unsqueeze(0).expand(B, -1, -1) # (5, 34, 16)

        positions_expanded = indices.unsqueeze(-1).expand(-1, -1, C)
        pos_emb = pos_emb_expanded * positions_expanded # (5, 34, 16)

        out = tok_emb + pos_emb # (5, 34, 16)
        out = self.blocks(out) # (5, 34, 16)
        out = self.linear_comb(out) # (5, 34, 1)
        # Convert to (5, 34)
        out = out.squeeze(2)
        out = self.ln1(out) # (5, 34)

        x = self.dropout(F.relu(self.linear1(out))) # (5, 256)
        x = self.linear2(x) # (5, 97)
        return x

    def get_attention_matrix(self, indices, sparse_vals):
        # print("Calculating attention matrix...")
        # px_cat = F.relu(self.token_embeddings_table(px_cat))
        # px_cont = F.relu(self.p_continous_embedding(px_cont))
        # mbx_cont = F.relu(self.mb_continuous_embedding(mbx_cont))
        # x_combined = torch.cat([px_cat, px_cont, mbx_cont], dim=1)
        # out = self.blocks.get_attention_matrix(x_combined)
        # return out
        pass


# Ramak

class PangenomeWindowedTransformerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = cfg.model.model_params.vocab_size # 50
        n_embd = cfg.model.model_params.n_embd # 32
        n_head = cfg.model.model_params.n_head # 4
        dropout = cfg.model.model_params.dropout # 0.1
        n_layer = cfg.model.model_params.n_layer # 2
        hidden_size = cfg.model.model_params.hidden_size # 32
        num_classes = cfg.preprocessing.dataset.num_classes # 97
        block_size = 50

        self.indices_embeddings_table = nn.Embedding(vocab_size, n_embd)
        self.sparse_vals_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = BlocksSequential(n_embd, n_head, dropout, n_layer)
        self.linear1 = nn.Linear(block_size*n_embd, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.fixed_indices = torch.arange(0, block_size, device=self.device)

    def forward(self, sparse_vals): #(5, 34), (5, 34)
        indices_emb = self.indices_embeddings_table(self.fixed_indices) # (5, 50, 32)
        sparse_vals_emb = self.sparse_vals_embedding_table(sparse_vals) # (5, 50, 32)

        out = indices_emb + sparse_vals_emb # (5, 50, 32)

        out = self.blocks(out) # (5, 50, 32) -> (5, 50*32) -> (5, 256) -> (5, 97)
        out = out.view(out.size(0), -1) # (5, 50*32)

        x = F.relu(self.linear1(out)) # (5, 256)
        x = self.linear2(x) # (5, 97)
        return x