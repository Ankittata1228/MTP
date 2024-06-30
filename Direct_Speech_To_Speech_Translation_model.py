import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import csv
from typing import Optional, Dict, List, Any, Union
import json
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import soundfile as sf


# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the Dictionary class
class Dictionary:
    def __init__(self, bos="<s>", pad="<pad>", eos="</s>", unk="<unk>", extra_special_symbols=None, add_special_symbols=True):
        self.bos_word, self.unk_word, self.pad_word, self.eos_word = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        if add_special_symbols:
            self.bos_index = self.add_symbol(bos)
            self.pad_index = self.add_symbol(pad)
            self.eos_index = self.add_symbol(eos)
            self.unk_index = self.add_symbol(unk)
            if extra_special_symbols:
                for s in extra_special_symbols:
                    self.add_symbol(s)
            self.nspecial = len(self.symbols)

    def add_symbol(self, unit, n=1, overwrite=False):
        if unit in self.indices and not overwrite:
            idx = self.indices[unit]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[unit] = idx
            self.symbols.append(unit)
            self.count.append(n)
            return idx

    def index(self, unit):
        return self.indices.get(unit, self.unk_index)

    def pad(self):
        return self.pad_index

    def eos(self):
        return self.eos_index

    def unk(self):
        return self.unk_index

    def save(self, filepath):
        data = {
            'symbols': self.symbols,
            'count': self.count,
            'indices': self.indices
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Dictionary saved to {filepath}")

    def __str__(self):
        return str({symbol: self.index(symbol) for symbol in self.symbols})

# Function to load audio and convert to mel spectrogram
def load_audio_and_get_spectrogram(audio_path, target_sample_rate=16000):
    waveform, sample_rate = librosa.load(audio_path, sr=target_sample_rate)
    spectrogram = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_fft=400, hop_length=160, n_mels=80, power=2.0)
    spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return torch.tensor(spectrogram, dtype=torch.float32)

# Custom Dataset
class SpeechToUnitDataset(Dataset):
    def __init__(self, tsv_file, dictionary, max_samples=10):
        self.data = []
        self.tgt_dict = dictionary
        self.max_length = 0
        with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t')
            for i, row in enumerate(reader):
                if i >= max_samples:
                    break
                src_audio_path = row['src_audio']
                spectrogram = load_audio_and_get_spectrogram(src_audio_path)
                target_units = row['tgt_audio'].split()
                self.data.append((spectrogram, target_units, src_audio_path))
                self.max_length = max(self.max_length, len(target_units))
                for unit in target_units:
                    self.tgt_dict.add_symbol(unit)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        spectrogram, target_units, src_audio_path = self.data[idx]
        target_indices = [self.tgt_dict.index(unit) for unit in target_units]
        return spectrogram, torch.tensor(target_indices, dtype=torch.long), src_audio_path

# Collate function to properly handle the padding of sequences
def collate_fn(batch):
    spectrograms, target_indices, src_audio_paths = zip(*batch)
    
    # Find the maximum length of the spectrograms in the batch
    max_spec_length = max(spectrogram.shape[1] for spectrogram in spectrograms)
    
    # Pad each spectrogram to the maximum length
    padded_spectrograms = [torch.nn.functional.pad(spectrogram, (0, max_spec_length - spectrogram.shape[1])) for spectrogram in spectrograms]
    spectrograms = torch.stack(padded_spectrograms)

    # Find the maximum length of the target sequences in the batch
    target_lengths = [len(indices) for indices in target_indices]
    max_target_length = max(target_lengths)
    
    # Pad each target sequence to the maximum length
    padded_targets = torch.full((len(batch), max_target_length), fill_value=batch[0][1].new_tensor(dataset_train.tgt_dict.pad()), dtype=torch.long)
    for i, indices in enumerate(target_indices):
        padded_targets[i, :len(indices)] = indices
        
    return spectrograms, padded_targets, src_audio_paths

# Function to create dataset and dataloader
def create_dataset_and_dataloader(tsv_file, dictionary, max_samples, batch_size=8):
    dataset = SpeechToUnitDataset(tsv_file, dictionary, max_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataset, dataloader

# Define the paths to your TSV files and save paths for the dictionaries
train_tsv_file_path = "/home/taruntejaneurips23/DEMO/final/final_train.tsv"
valid_tsv_file_path = "/home/taruntejaneurips23/DEMO/final/final_valid.tsv"
train_dict_save_path = "/home/taruntejaneurips23/DEMO/final_train_dict.json"
valid_dict_save_path = "/home/taruntejaneurips23/DEMO/final_valid_dict.json"

# Create dictionaries
train_dictionary = Dictionary()
valid_dictionary = Dictionary()

# Create datasets and dataloaders with a smaller batch size
dataset_train, dataloader_train = create_dataset_and_dataloader(train_tsv_file_path, train_dictionary, max_samples=200, batch_size=1) 
dataset_valid, dataloader_valid = create_dataset_and_dataloader(valid_tsv_file_path, valid_dictionary, max_samples=200, batch_size=1)  

# Save dictionaries
train_dictionary.save(train_dict_save_path)
valid_dictionary.save(valid_dict_save_path)

class ConvolutionalDownsampling(nn.Module):
    def __init__(self, input_channels, model_dim, dropout_rate=0.12):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, model_dim, kernel_size=5, stride=2, padding=2)
        self.gelu1 = nn.GELU()
        self.norm1 = nn.BatchNorm1d(model_dim)
        self.conv2 = nn.Conv1d(model_dim, model_dim, kernel_size=5, stride=2, padding=2)
        self.gelu2 = nn.GELU()
        self.norm2 = nn.BatchNorm1d(model_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

    def forward(self, x):
        if x.dim() == 2:  
            x = x.unsqueeze(1)  # Add channel dimension to [batch_size, 1, seq_len]
        if x.shape[1] != 80:
            x = x.permute(0, 2, 1)  # Convert to [batch_size, features, seq_len] for Conv1D
        x = self.dropout1(self.gelu1(self.norm1(self.conv1(x))))
        x = self.dropout1(self.gelu2(self.norm2(self.conv2(x))))
        x = x.permute(0, 2, 1)  # Convert back to [batch_size, seq_len, features]
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, nhead, dropout_rate=0.1, normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(model_dim, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, model_dim)
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before

    def forward(self, src):
        if self.normalize_before:
            src = self.norm1(src)
        attn_output, _ = self.self_attn(src, src, src)
        src = src + self.dropout(attn_output)
        if not self.normalize_before:
            src = self.norm1(src)
        if self.normalize_before:
            src = self.norm2(src)
        src = src + self.dropout(self.feed_forward(src))
        if not self.normalize_before:
            src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_layers, nhead, dropout_rate=0.1, layerdrop_rate=0.0, normalize_before=False):
        super().__init__()
        self.downsampling = ConvolutionalDownsampling(input_dim, model_dim, dropout_rate)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(model_dim, nhead, dropout_rate, normalize_before)
            for _ in range(num_layers)
        ])
        self.layerdrop_rate = layerdrop_rate
        self.layer_norm = nn.LayerNorm(model_dim) if normalize_before else None
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, src):
        src = self.downsampling(src)
        src = self.dropout(src)
        outputs = []
        for i, layer in enumerate(self.layers):
            if self.training and (torch.rand(1).item() < self.layerdrop_rate):
                continue
            src = layer(src)
            outputs.append(src)
        if self.layer_norm:
            src = self.layer_norm(src)
            
            #print(outputs.shape)
        #print("src shape",src.shape)
        return {
            "last_layer_output": src,  
            "all_layer_outputs": outputs 
        }

class StackedEmbedding(nn.Embedding):
    """Embedding module that supports stacked units -> single embedding"""

    def __init__(self, num_embeddings, embed_dim, padding_idx, num_stacked=1):
        super().__init__(num_embeddings, embed_dim, padding_idx)
        nn.init.normal_(self.weight, mean=0, std=embed_dim ** -0.5)
        nn.init.constant_(self.weight[padding_idx], 0)

        self.offset = 4  # skip <bos>, <pad>, <eos>, <unk>, specific to fairseq dictionary
        self.vocab_size = num_embeddings - self.offset
        self.num_stacked = num_stacked

        if self.num_stacked > 1:
            self.project_in_dim = Linear(embed_dim * num_stacked, embed_dim, bias=False)

    def forward(self, input):
        if self.num_stacked == 1:
            return super().forward(input)
        mask = input >= self.offset
        stacked_input = []
        cum_input = input.new_zeros(input.shape)
        for i in range(1, self.num_stacked + 1):
            div = pow(self.vocab_size, i)
            next_input = torch.remainder(input - self.offset - cum_input, div)
            cum_input += next_input
            next_input = torch.floor_divide(next_input, div // self.vocab_size)
            stacked_input.append((next_input + self.offset) * mask + input * ~mask)

        stacked_input = torch.stack(stacked_input[::-1], dim=2)
        embed = super().forward(stacked_input).view(input.size(0), input.size(1), -1)
        embed = self.project_in_dim(embed)
        return embed

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m

class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, activation='relu', normalize_before=False):
        super(TransformerDecoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.encoder_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.activation_fn = F.relu if activation == 'relu' else F.gelu
        self.dropout = nn.Dropout(dropout)

    def _add_residual(self, residual, x):
        return residual + x

    def forward(
        self,
        x,
        encoder_out: Optional[Union[Dict[str, torch.Tensor], torch.Tensor]] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ):
        #print("x input in forward of transformer decoder layer",x.shape)
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=self_attn_mask,
            key_padding_mask=self_attn_padding_mask,
        )
        x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)

        x = self.dropout(x)
        x = self._add_residual(residual, x)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if encoder_out is not None:
            # Handle both tensor and dictionary cases for encoder_out
            if isinstance(encoder_out, dict):
                encoder_out_tensor = encoder_out["last_layer_output"]
            else:
                encoder_out_tensor = encoder_out
        
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

            x = x.transpose(0, 1)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
            encoder_out_tensor = encoder_out_tensor.transpose(0, 1)  # (batch_size, encoder_seq_len, embed_dim) -> (encoder_seq_len, batch_size, embed_dim)
    
            x, _ = self.encoder_attn(
                query=x,
                key=encoder_out_tensor,
                value=encoder_out_tensor,
                key_padding_mask=encoder_padding_mask,
                attn_mask=memory_mask,
            )
            x = x.transpose(0, 1)  # (seq_len, batch_size, embed_dim) -> (batch_size, seq_len, embed_dim)

            x = self.dropout(x)
            x = self._add_residual(residual, x)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self._add_residual(residual, x)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        #print("decoder layer output x",x.shape)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_tokens, embed_dim, nhead, dropout=0.1, num_stacked=1, padding_idx=0, max_seq_len=600 ):
        super().__init__()
        self.embedding = embed_tokens
        self.embed_positions = nn.Embedding(max_seq_len, embed_dim)
        self.layers = nn.ModuleList([TransformerDecoderLayer(embed_dim, nhead, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.output_projection = nn.Linear(embed_dim, self.embedding.num_embeddings)
        self.dropout = nn.Dropout(dropout)
        self._future_mask = torch.empty(0)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                torch.full((dim, dim), float('-inf'), device=tensor.device), 1
            )
        return self._future_mask[:dim, :dim]

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, target=None):
        #print("tgt in transformer decoder",tgt)
        positions = torch.arange(tgt.size(1), device=tgt.device).unsqueeze(0)
        tgt = self.embedding(tgt)
        #print("tgt in forward transformer decoder layer",tgt.shape)
        tgt += self.embed_positions(positions)
        tgt = self.dropout(tgt)

        # Generate future mask if not provided
        if tgt_mask is None:
            tgt_mask = self.buffered_future_mask(tgt)
            #print("tgt mask",tgt_mask)
        for i, layer in enumerate(self.layers):
            tgt = layer(
                tgt,
                memory,
                self_attn_mask=tgt_mask,
                memory_mask=memory_mask,
                self_attn_padding_mask=tgt_key_padding_mask,
                encoder_padding_mask=memory_key_padding_mask
            )

        tgt = self.norm(tgt)
        tgt = self.output_projection(tgt)
        return tgt

class TransformerUnitDecoder(nn.Module):
    def __init__(self, num_layers, dictionary, embed_tokens, embed_dim, nhead, dropout=0.1, num_stacked=1, padding_idx=0, n_frames_per_step=1):
        super().__init__()
        self.dictionary = dictionary
        self.embed_tokens = StackedEmbedding(
            num_embeddings=len(dictionary.symbols),
            embed_dim=embed_dim,
            padding_idx=padding_idx,
            num_stacked=num_stacked
        )
        self.n_frames_per_step = n_frames_per_step
        self.decoder = TransformerDecoder(num_layers, self.embed_tokens, embed_dim, nhead, dropout, num_stacked, padding_idx)
        self.out_proj_n_frames = nn.Linear(embed_dim, embed_dim * n_frames_per_step, bias=False) if n_frames_per_step > 1 else None

    def forward(self, prev_output_tokens, encoder_out: Optional[Dict[str, List[torch.Tensor]]] = None, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None, features_only: bool = False, full_context_alignment: bool = False, alignment_layer: Optional[int] = None, alignment_heads: Optional[int] = None, src_lengths: Optional[Any] = None, return_all_hiddens: bool = False, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        #print("prev_output_tokens",prev_output_tokens)
        x, extra = self.extract_features(prev_output_tokens, encoder_out=encoder_out, incremental_state=incremental_state, full_context_alignment=full_context_alignment, alignment_layer=alignment_layer, alignment_heads=alignment_heads, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        #print("x output of extract features",x.shape)
        if not features_only:
            bsz, seq_len, d = x.size()
            if self.out_proj_n_frames:
                x = self.out_proj_n_frames(x)
            x = self.decoder.output_projection(x.view(bsz, seq_len, self.n_frames_per_step, d))
            x = x.view(bsz, seq_len * self.n_frames_per_step, -1)
            if incremental_state is None and self.n_frames_per_step > 1:  # teacher-forcing mode in training
                x = x[:, : -(self.n_frames_per_step - 1), :]  # remove extra frames after <eos>
        return x, extra

    def extract_features(self, prev_output_tokens, encoder_out: Optional[Dict[str, List[torch.Tensor]]] = None, incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None, full_context_alignment: bool = False, alignment_layer: Optional[int] = None, alignment_heads: Optional[int] = None, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        #print("prev_output_tokens in extract features",prev_output_tokens)
        positions = torch.arange(prev_output_tokens.size(1), device=prev_output_tokens.device).unsqueeze(0)
        # Ensure positions are within the embedding range
        positions = positions.clamp(0, self.decoder.embed_positions.num_embeddings - 1)
        x = self.decoder.embedding(prev_output_tokens)
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            positions = positions[:, -1:]
        if positions.size(1) > x.size(1):
            positions = positions[:, :x.size(1)]
        x += self.decoder.embed_positions(positions)
        x = nn.functional.dropout(x, p=0.1, training=self.training)
        attn: Optional[torch.Tensor] = None
        inner_states: List[Optional[torch.Tensor]] = [x]
        for idx, layer in enumerate(self.decoder.layers):
            x = layer(
                x,
                encoder_out if encoder_out is not None else None,
                self_attn_mask=tgt_mask,
                memory_mask=memory_mask,
                self_attn_padding_mask=tgt_key_padding_mask,
                encoder_padding_mask=memory_key_padding_mask
            )
            inner_states.append(x)
        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]
            attn = attn.mean(dim=0)
        #print("x output of unit decoder",x)
        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        return self.decoder.output_projection(features)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(1)
        return torch.triu(torch.full((dim, dim), float('-inf')), 1)

    def upgrade_state_dict_named(self, state_dict, name):
        if self.n_frames_per_step > 1:
            move_keys = [
                (
                    f"{name}.project_in_dim.weight",
                    f"{name}.embed_tokens.project_in_dim.weight",
                )
            ]
            for from_k, to_k in move_keys:
                if from_k in state_dict and to_k not in state_dict:
                    state_dict[to_k] = state_dict[from_k]
                    del state_dict[from_k]

class SpeechToUnitModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        memory = self.encoder(src)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        return output

# Define paths and parameters
checkpoint_dir = "/home/taruntejaneurips23/DEMO/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
num_epochs = 70
learning_rate = 0.0001

# Model, criterion, optimizer
embed_dim = 256
num_heads = 100
dropout = 0.1
num_layers = 6
input_dim = 80
model_dim = 256
encoder_layers = 10
decoder_layers = 6
encoder_nhead = 4
decoder_nhead = 8
vocab_size = 104

num_embeddings = 104

padding_idx = 0
n_frames_per_step = 1

dictionary = dataset_valid.tgt_dict
embed_tokens = StackedEmbedding(
    num_embeddings,
    embed_dim,
    padding_idx,
    num_stacked=n_frames_per_step,
)

encoder = TransformerEncoder(input_dim, model_dim, encoder_layers, encoder_nhead)
decoder = TransformerUnitDecoder(decoder_layers, dictionary, embed_tokens, embed_dim, decoder_nhead)
model = SpeechToUnitModel(encoder, decoder)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Check for GPU availability
device_id = 1  # Change this to the desired CUDA device ID
torch.cuda.set_device(device_id)
print(f"Using device: {device}")

# Function to save the model state
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Model saved to {filepath}")

def plot_and_save_losses(train_losses, valid_losses, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()
    print(f"Loss plot saved to {filename}")

# Training function
def train_model(model, dataloader_train, dataloader_valid, criterion, optimizer, num_epochs, checkpoint_dir, device):
    model.to(device)
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(dataloader_train, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            spectrograms, target_units ,source_audio_paths= batch
            spectrograms, target_units = spectrograms.to(device), target_units.to(device)
            outputs, _ = model(spectrograms, target_units)
            num_classes = outputs.size(-1)
            target_units = target_units.clamp(0, num_classes - 1).view(-1)
            #print("target_units",target_units)
            outputs = outputs.reshape(-1, num_classes)        
            loss = criterion(outputs, target_units)
            outputs = nn.functional.softmax(outputs, dim=-1)
            
            discrete_units = torch.argmax(outputs, dim=-1)
            #print("discrete_units" ,discrete_units )
            loss.backward()
           
            # Add gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(dataloader_train)
        train_losses.append(avg_train_loss)
        print(f"Training Loss: {avg_train_loss}")

        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in dataloader_valid:
                spectrograms, target_units ,source_audio_paths= batch
                spectrograms, target_units = spectrograms.to(device), target_units.to(device)
                outputs, _ = model(spectrograms,target_units)
                num_classes = outputs.size(-1)
                target_units = target_units.clamp(0, num_classes - 1).view(-1)
                #print("target_units",target_units)
                outputs = outputs.reshape(-1, num_classes)
                loss = criterion(outputs, target_units)
                outputs = nn.functional.softmax(outputs, dim=-1)
                #print("output",outputs)
                discrete_units = torch.argmax(outputs, dim=-1)
                #print("discrete_units",discrete_units)
                valid_loss += loss.item()
                
        avg_valid_loss = valid_loss / len(dataloader_valid)
        valid_losses.append(avg_valid_loss)
        print(f"Validation Loss: {avg_valid_loss}")
    
        # Save the best model checkpoint based on validation loss
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Best model checkpoint saved to {best_checkpoint_path} with validation loss: {best_valid_loss}")
    plot_and_save_losses(train_losses, valid_losses, os.path.join(checkpoint_dir, 'loss_plot.png'))

    return train_losses, valid_losses
#train_model(model, dataloader_train, dataloader_valid, criterion, optimizer, num_epochs, checkpoint_dir, device)



# Function to generate and save discrete units with target units
def generate_and_save_units(model, dataloader_valid, dictionary, device, output_dir):
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    with torch.no_grad():
        for i, batch in enumerate(dataloader_valid):
            spectrograms, target_units, src_audio_paths = batch
            spectrograms, target_units = spectrograms.to(device), target_units.to(device)
            outputs, _ = model(spectrograms, target_units)

            num_classes = outputs.size(-1)
            target_units = target_units.clamp(0, num_classes - 1).view(-1)
            outputs = outputs.view(-1, num_classes)

            outputs = F.softmax(outputs, dim=-1)
            discrete_units = torch.argmax(outputs, dim=-1)


            discrete_symbols = [dictionary.symbols[idx] for idx in discrete_units.cpu().numpy()]


            discrete_symbols_filtered = [symbol for symbol in discrete_symbols if symbol != '<pad>']

            # Save to file
            for src_audio_path in src_audio_paths:
                audio_filename = os.path.basename(src_audio_path)
                unit_sequence_filename = os.path.splitext(audio_filename)[0] + '_units.txt'
                unit_sequence_path = os.path.join(output_dir, unit_sequence_filename)
                with open(unit_sequence_path, 'w') as f:
                    f.write(' '.join(discrete_symbols_filtered) + '\n')

                #print(f"Unit sequence saved to {unit_sequence_path}")

# Define paths and parameters
checkpoint_path = "/home/taruntejaneurips23/DEMO/checkpoints/best_model_80.pth"
output_dir = "/home/taruntejaneurips23/DEMO/generated_units_1"

# Function to load the model state from a checkpoint
def load_model(model, checkpoint_path, device):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {checkpoint_path}")
    return model

def load_dictionary(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        dictionary = Dictionary()
        dictionary.symbols = data['symbols']
        dictionary.count = data['count']
        dictionary.indices = data['indices']
    return dictionary

valid_dict_path = "/home/taruntejaneurips23/DEMO/final_valid_dict.json"
valid_dictionary = load_dictionary(valid_dict_path)
print(f"Loaded dictionary with {len(valid_dictionary.symbols)} symbols.")

# Load the trained model from the checkpoint
model = load_model(model, checkpoint_path, device)

# Generate and save discrete units with target units for validation samples
generate_and_save_units(model, dataloader_valid, valid_dictionary, device, output_dir)


acoustic = torch.hub.load("bshall/acoustic-model:main", "hubert_discrete", trust_repo=True).cuda()
hifigan = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_discrete", trust_repo=True).cuda()

# Function to read units from a text file
def read_units_from_file(file_path):
    with open(file_path, 'r') as f:
        units = f.read().strip().split()
        units = [int(unit) for unit in units]
    return units

# Path to the directory containing units files
units_dir = "/home/taruntejaneurips23/DEMO/generated_units"
output_dir = "/home/taruntejaneurips23/DEMO/generated_audio_1"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate over all unit files in the directory
for units_file in os.listdir(units_dir):
    if units_file.endswith('.txt'):
        units_file_path = os.path.join(units_dir, units_file)
        
        # Read units from the file
        units = read_units_from_file(units_file_path)
        units = torch.tensor(units).unsqueeze(0).cuda()  
        #print("Units shape:", units.shape)
        
        # Generate target spectrogram using the acoustic model
        mel = acoustic.generate(units)
        #print("Mel spectrogram shape:", mel.shape)

        
        #mel = mel.clone()

        mel = mel.permute(0, 2, 1)  
        #print("Processed mel spectrogram shape:", mel.shape)
        


        # Generate audio waveform using the HiFi-GAN vocoder
        with torch.no_grad():
            audio_output = hifigan(mel)
        audio_output = audio_output.squeeze()
        #print("Generated audio shape:", audio_output.shape)

        # Extract the base filename 
        base_filename = os.path.splitext(units_file)[0]
        
        # Save the generated audio 
        output_audio_path = os.path.join(output_dir, f"{base_filename}.wav")
        sf.write(output_audio_path, audio_output.cpu().numpy(), 16000)
        print(f"Audio saved to {output_audio_path}")



