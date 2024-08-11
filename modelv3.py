import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device :", device)

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # alpha is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # bias is a learnable parameter

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term_even = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        if (d_model % 2 != 0):
            div_term_odd = torch.exp(torch.arange(0, d_model-2, 2).float() * (-math.log(10000.0) / d_model))
        else:
            div_term_odd = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[2], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class EEGformerEncoder(nn.Module):
    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.linear = nn.Linear(features, features, bias=True)
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        flat_input = inputs.view(inputs.shape[0],-1, self.embedding_dim)

        distance = (torch.sum(flat_input**2, dim=2, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        encoding_indices = torch.argmin(distance, dim=2).unsqueeze(2)
        encodings = torch.zeros(flat_input.shape[0], encoding_indices.shape[1], self.num_embeddings, device=inputs.device)
        encodings.scatter(2, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = inputs + (quantized-inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return quantized, loss, perplexity

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class EEGformerDecoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim, d_model, num_cls) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)  # Meratakan dimensi (dim1, dim2) menjadi satu dimensi
        self.linear1 = nn.Linear(input_dim * d_model, input_dim * d_model *4)  # Proyeksi pertama
        self.linear2 = nn.Linear(input_dim * d_model*4, num_cls)  # Proyeksi kedua

      
    def forward(self, x):
        x = self.flatten(x)  
        x = self.linear1(x)  
        x = self.linear2(x)  
        return torch.softmax(x, dim=1)

class EEGformer(nn.Module):
    def __init__(self,
                 pos_embed : PositionalEncoding,
                 encoder: EEGformerEncoder,
                 vec_quantizer: VectorQuantizer,
                 decoder: EEGformerDecoder,
                 projector: ProjectionLayer
                 ) -> None:
        super().__init__()
        self.pos_embed = pos_embed
        self.encoder = encoder
        self.vec_quantizer = vec_quantizer
        self.decoder = decoder
        self.projector = projector

    def embed_pos(self, x:torch.Tensor):
        return self.pos_embed(x)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)
      
    def code_book(self, x:torch.Tensor):
        return self.vec_quantizer(x)

    def decode(self, encoder_output: torch.Tensor, tgt:torch.Tensor):
        return self.decoder(encoder_output, tgt)
    
    def project(self, x:torch.tensor):
        return self.projector(x)


def build_eegformer(
       input_dim=9,
       d_model=128,
       h=8,
       N=3,
       num_embeddings=1024,
       commitment_cost=0.25,
       epoch=100,
       dropout=0.1,
       num_cls=5
       ):
    
    

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block  = MultiHeadAttentionBlock(d_model, h, dropout)
        encoder_feed_froward_block            = FeedForwardBlock(d_model, d_model * 4, dropout)
        encoder_block        = EncoderBlock(d_model, encoder_self_attention_block, encoder_feed_froward_block, dropout)
        encoder_blocks.append(encoder_block)
    
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block  = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block  = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_froward_block            = FeedForwardBlock(d_model, d_model * 4, dropout)
        decoder_block= DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_froward_block, dropout)
        decoder_blocks.append(decoder_block)

    positional_encoding = PositionalEncoding(d_model, input_dim, dropout)
    encoder = EEGformerEncoder(d_model, nn.ModuleList(encoder_blocks))
    vector_quantizer = VectorQuantizer(num_embeddings, d_model, commitment_cost)
    decoder = EEGformerDecoder(d_model, nn.ModuleList(decoder_blocks))
    projector = ProjectionLayer(input_dim, d_model, num_cls)

    
    eegformer = EEGformer(positional_encoding, encoder, vector_quantizer, decoder, projector)
    
    for p in eegformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  
    
    return eegformer