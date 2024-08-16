import torch
import torch.nn as nn
import math

class OneDCNN(nn.Module):
    def __init__(self, in_chan, feature, dtype=torch.float32):
        super().__init__()
        self.in_chan = in_chan
        self.dtype = dtype
        self.feature = feature

        self.conv1 = nn.Conv1d(self.in_chan, self.in_chan, kernel_size=10, stride=1, padding='valid', groups=self.in_chan, dtype=self.dtype)
        self.conv2 = nn.Conv1d(self.in_chan, self.in_chan, kernel_size=10, stride=1, padding='valid', groups=self.in_chan, dtype=self.dtype)
        self.conv3 = nn.Conv1d(self.in_chan, self.feature * self.in_chan, kernel_size=10, stride=1, padding='valid', groups=self.in_chan, dtype=self.dtype)
        self.gelu = nn.GELU()

    def forward(self, x):
        # (Batch, Channel, Seq_Len)
        x = self.conv1(x)
        x = self.gelu(x)

        x = self.conv2(x)
        x = self.gelu(x)

        x = self.conv3(x)
        x = self.gelu(x)

        x = torch.reshape(x, (x.shape[0], (int)(x.shape[1]/self.feature), self.feature, (int)(x.shape[2])))
        return x

class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

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

        query = query.view(query.shape[0], query.shape[1], query.shape[2], self.h, self.d_k).transpose(2,3)
        key = key.view(key.shape[0], key.shape[1], key.shape[2], self.h, self.d_k).transpose(2,3)
        value = value.view(value.shape[0], value.shape[1], value.shape[2], self.h, self.d_k).transpose(2,3)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        x = x.transpose(2, 3).contiguous().view(x.shape[0], x.shape[1], -1, self.h * self.d_k)
        return self.w_o(x)

class MultiHeadAttentionBlockTemporal(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
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

class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: 'MultiHeadAttentionBlock | MultiHeadAttentionBlockTemporal', feed_forward_block: FeedForwardBlock, dropout: float) -> None:      
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class RegionalEncoder(nn.Module):
    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.linear = nn.Linear(features, features, bias=True)
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class SyncEncoder(nn.Module):
    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.linear = nn.Linear(features, features, bias=True)
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class TemporalEncoder(nn.Module):
    def __init__(self, features:int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.linear = nn.Linear(features, features, bias=True)
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class EEGformerEncoder(nn.Module):
    def __init__(
        self,
        regional_encoder: RegionalEncoder,
        sync_encoder: SyncEncoder,
        temporal_encoder: TemporalEncoder,
        onedcnn_seq: int,
        patch_region: int,
        patch_sync: int,
        submetric_temporal: int,
        ) -> None:
        super().__init__()
        self.onedcnn_seq = onedcnn_seq
        self.submetric_temporal = submetric_temporal

        self.patch_region = patch_region
        self.patch_sync = patch_sync
        self.patch_temporal = submetric_temporal + 1

        self.submetric_size = patch_region * patch_sync

        self.linear_before_regional = nn.Linear(onedcnn_seq, onedcnn_seq)
        self.positional_encoding_regional = PositionalEncoding(onedcnn_seq, self.patch_region, 0.1)

        self.linear_before_sync = nn.Linear(onedcnn_seq, onedcnn_seq)
        self.positional_encoding_sync = PositionalEncoding(onedcnn_seq, self.patch_sync, 0.1)

        self.linear_before_temporal = nn.Linear(self.submetric_size,self.submetric_size)
        self.positional_encoding_temporal = PositionalEncoding(self.submetric_size, self.patch_temporal, 0.1)


        self.regional_encoder = regional_encoder
        self.sync_encoder = sync_encoder
        self.temporal_encoder = temporal_encoder

    def forward(self, x):
        # REGIONAL PART
        # Class token
        class_token_regional = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3]).to(x.device)
        x = torch.cat((class_token_regional, x), dim=2)
        # Linear Mapping
        x = self.linear_before_regional(x)
        # Positional Embedding here
        x = self.positional_encoding_regional(x)
        x = self.regional_encoder(x, None)
        # print("Regional Out shape : ", x.shape)

        # SYNCHRONIZATION PART
        # Transpose x
        x = x.transpose(1, 2)
        # Class token
        class_token_sync = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3]).to(x.device)
        x = torch.cat((class_token_sync, x), dim=2)
        # Linear Mapping
        x = self.linear_before_sync(x)
        # Positional Embedding here
        x = self.positional_encoding_sync(x)
        x = self.sync_encoder(x, None)
        # print("Sync Out shape : ", x.shape)

        # TEMPORAL PART
        # Transpose x
        x = x.transpose(1,3)
        # Break to Submatrix and count AVG
        x = x.view(x.shape[0], self.onedcnn_seq//self.submetric_temporal, self.submetric_temporal, x.shape[2], x.shape[3])
        x = x.mean(dim=1)
        # Flatten
        x = x.flatten(start_dim=-2)
        # Class token
        class_token_temp = torch.zeros(x.shape[0], 1, x.shape[2]).to(x.device)
        x = torch.cat((class_token_temp, x), dim=1)
        # Linear Mapping untuk dapat latent z
        x = self.linear_before_temporal(x)
        # Positional Embedding here
        x = self.positional_encoding_temporal(x)
        x = self.temporal_encoder(x, None)
        # print("Temporal Out shape : ", x.shape)
        return x

class EEGformerDecoder(nn.Module):
    def __init__(self, features_s: int, features_c: int, features_m: int, hidden_features: int, num_cls: int, patch_regional:int, patch_sync:int) -> None:
        super().__init__()
        self.patch_regional = patch_regional
        self.patch_sync = patch_sync
        self.num_cls = num_cls
        self.hidden_features = hidden_features

        self.conv_1 = nn.Conv1d(features_c, 1, kernel_size=1)
        self.conv_2 = nn.Conv1d(features_s, hidden_features, kernel_size=1)
        self.conv_3 = nn.Conv1d(features_m, int(features_m/2), kernel_size=1)
        self.proj = nn.Linear(int(features_m/2) * hidden_features, num_cls)
        self.gelu = nn.GELU()

    def forward(self, x):
        # print("INSIDE DECODER")

        x = x.reshape(x.shape[0], x.shape[1], self.patch_sync, self.patch_regional) # 8,11,4,121
        init_shape = x.shape
        # print("Init : ", init_shape)
        x = x.reshape(init_shape[0] * init_shape[1], self.patch_sync, self.patch_regional).transpose(1,2) # 88, 4,121
        # print(x.shape)
        x = self.conv_1(x)
        x = self.gelu(x)
        x = x.transpose(1,2)
        # print(x.shape)
        x = self.conv_2(x)
        x = self.gelu(x)
        x = x.reshape(init_shape[0], init_shape[1], x.shape[1], x.shape[2])
        x = x.transpose(1,2)
        x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        # print(x.shape)
        x = self.conv_3(x)
        x = self.gelu(x)
        x = x.reshape(init_shape[0], self.hidden_features, self.num_cls, 1)
        x = x.reshape(init_shape[0], self.hidden_features * self.num_cls, 1)
        x = x.reshape(init_shape[0], self.hidden_features * self.num_cls)
        # print(x.shape)
        x = self.proj(x)
        # print(x.shape)
        x = torch.softmax(x, dim=1)
        return x

class EEGformer(nn.Module):
    def __init__(self,
                 onedcnn: OneDCNN,
                 encoder: EEGformerEncoder,
                 decoder: EEGformerDecoder
                 ) -> None:
        super().__init__()
        self.onedcnn = onedcnn
        self.encoder = encoder
        self.decoder = decoder

    def construct3D(self, x):
        return self.onedcnn(x)

    def encode(self, x: torch.Tensor):
        return self.encoder(x)

    def decode(self, encoder_output: torch.Tensor):
        return self.decoder(encoder_output)

def build_eegformer(
        channel_size: int,
        seq_len: int,
        N: int = 3,
        feature_onedcnn: int=120,
        kernel_size: int=10,
        h_regional: int=5,
        h_sync: int=5,
        h_temp: int=11,
        dropout:float = 0.1,
        sub_matrices:int = 10,
        feature_decoder:int = 2,
        num_cls: int=5,
        scaler_ffn: int=4):

    # Calculate some feature size
    conv_temporal = seq_len - 3 * (kernel_size - 1) # 350 = 377 - 27
    assert conv_temporal % sub_matrices == 0, "Temporal Sequence (conv_temporal) is not divisible by Sub Matrices (sub_matrices).\nCheck length of EEG sequence after processed by OneDCNN"

    patch_regional = feature_onedcnn + 1
    patch_sync =  channel_size + 1
    patch_temporal = sub_matrices + 1

    map_f_channel = patch_regional * patch_sync

    onedcnn = OneDCNN(channel_size, feature_onedcnn)
    # Output Shape OneDCNN without Batch

    regional_encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block  = MultiHeadAttentionBlock(conv_temporal, h_regional, dropout)
        feed_froward_block            = FeedForwardBlock(conv_temporal, conv_temporal * scaler_ffn, dropout)
        regional_encoder_block        = EncoderBlock(conv_temporal, encoder_self_attention_block, feed_froward_block, dropout)
        regional_encoder_blocks.append(regional_encoder_block)

    sync_encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block  = MultiHeadAttentionBlock(conv_temporal, h_sync, dropout)
        feed_froward_block            = FeedForwardBlock(conv_temporal, conv_temporal * scaler_ffn, dropout)
        sync_encoder_block            = EncoderBlock(conv_temporal, encoder_self_attention_block, feed_froward_block, dropout)
        sync_encoder_blocks.append(sync_encoder_block)

    temp_encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block  = MultiHeadAttentionBlockTemporal(map_f_channel, h_temp, dropout)
        feed_froward_block            = FeedForwardBlock(map_f_channel, map_f_channel * scaler_ffn, dropout)
        temp_encoder_block            = EncoderBlock(map_f_channel, encoder_self_attention_block, feed_froward_block, dropout)
        temp_encoder_blocks.append(temp_encoder_block)
    
    regional_encoder = RegionalEncoder(conv_temporal, nn.ModuleList(regional_encoder_blocks))
    sync_encoder = SyncEncoder(conv_temporal, nn.ModuleList(sync_encoder_blocks))
    temp_encoder = TemporalEncoder(map_f_channel, nn.ModuleList(temp_encoder_blocks))

    eegformer_encoder = EEGformerEncoder(regional_encoder, sync_encoder, temp_encoder, conv_temporal, patch_regional, patch_sync, sub_matrices)    
    eegformer_decoder = EEGformerDecoder(patch_sync, patch_regional, patch_temporal, feature_decoder, num_cls, patch_regional, patch_sync)                         
    eegformer = EEGformer(onedcnn, eegformer_encoder, eegformer_decoder)
    return eegformer