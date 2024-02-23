# %%
import torch
import os
torch.cuda.is_available()

import spacy

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

import pdb
# %%
from transformers import AutoTokenizer
from torchtext.data import Field, Example, Dataset
import numpy as np

import sentencepiece as spm
import multiprocessing

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

rank = int(os.environ['RANK'])
local_rank = int(os.environ['LOCAL_RANK'])
world_size = int(os.environ['WORLD_SIZE'])

# print(rank)
# print(local_rank)
# print(world_size)

def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
torch.cuda.set_device(local_rank)  
setup(rank, world_size)
dist.barrier()


# sentencepice training
#spm.SentencePieceTrainer.Train("--input=de.txt --model_prefix=de_model --vocab_size=30000 --model_type=bpe --max_sentence_length=9999 --pad_id=1 --unk_id=0 --bos_id=2 --eos_id=3")
#spm.SentencePieceTrainer.Train("--input=en.txt --model_prefix=en_model --vocab_size=30000 --model_type=bpe --max_sentence_length=9999 --pad_id=1 --unk_id=0 --bos_id=2 --eos_id=3")

#tokenizer = AutoTokenizer.from_pretrained('gpt2')

#sp_de = spm.SentencePieceProcessor()
#sp_en = spm.SentencePieceProcessor()

#sp_de.Load("de_model.model")
#sp_en.Load("en_model.model")

#huggingface tokenizer
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

def tokenize_de(text):
    return tokenizer.tokenize(text)
#     return [token.text for token in spacy_de.tokenizer(text)]


def tokenize_en(text):
    return tokenizer.tokenize(text)
#     return [token.text for token in spacy_en.tokenizer(text)]


english = "my name is john and i want to make some models."

l = tokenize_en(english)


# multi30k dataset
# import os

# raw_dir = "/jeongho/yezun/data/multi30k/raw"
# with open(os.path.join(raw_dir, "train.en"), "r") as f:
#     train_en = [text.rstrip() for text in f]
# with open(os.path.join(raw_dir, "train.de"), "r") as f:
#     train_de = [text.rstrip() for text in f]
# small_train_dataset = [(de, en) for de, en in zip(train_de, train_en)]

# with open(os.path.join(raw_dir, "val.en"), "r") as f:
#     valid_en = [text.rstrip() for text in f]
# with open(os.path.join(raw_dir, "val.de"), "r") as f:
#     valid_de = [text.rstrip() for text in f]
# small_valid_dataset = [(de, en) for de, en in zip(valid_de, valid_en)]

# with open(os.path.join(raw_dir, "test_2018_flickr.en"), "r") as f:
#     test_en = [text.rstrip() for text in f]
# with open(os.path.join(raw_dir, "test_2018_flickr.de"), "r") as f:
#     test_de = [text.rstrip() for text in f]
# small_test_dataset = [(de, en) for de, en in zip(test_de, test_en)]

# # Print the sizes of the subsets
# print("Small Train Size:", len(small_train_dataset))
# print("Small Valid Size:", len(small_valid_dataset))
# print("Small Test Size:", len(small_test_dataset))







# %%
from datasets import load_dataset

dataset = load_dataset("wmt14", "de-en")

new_train_size = 20
new_valid_size = 20
new_test_size = 20

#Split the dataset
small_train_dataset_wmt = dataset["train"].shuffle(seed=42).select(range(new_train_size))
small_valid_dataset_wmt = dataset["validation"].shuffle(seed=42).select(range(new_valid_size))
small_test_dataset_wmt = dataset["test"].shuffle(seed=42).select(range(new_test_size))



# small_train_dataset_wmt = dataset["train"]
# small_valid_dataset_wmt = dataset["validation"]
# small_test_dataset_wmt = dataset["test"]


print("Small Train Size:", len(small_train_dataset_wmt))
print("Small Valid Size:", len(small_valid_dataset_wmt))
print("Small Test Size:", len(small_test_dataset_wmt))

print(small_train_dataset_wmt[0]['translation']['en'])


# # Bucketiterator

# max_len = 500
# examples_train_wmt = []
# len_train = []
# for example in small_train_dataset_wmt['translation']:
#     de = example['de']
#     en = example['en']
#     if len(de) <= max_len and len(en) <= max_len:
#         len_train.append(len(de))
#         examples_train_wmt.append(Example.fromlist([de, en], fields=[('de', SRC), ('en', TRG)]))
#         if len(len_train) % 10000 == 0:
#             print(f"{len(len_train)} / {len(small_train_dataset_wmt)}")
# # while len(examples_train_wmt) % 12800 != 0:
# #     examples_train_wmt.pop()
# print(len(examples_train_wmt))

# train_data_wmt = Dataset(examples_train_wmt, fields=[('de', SRC), ('en', TRG)])

# # print(len_train)
# examples_valid_wmt = []
# for example in small_valid_dataset_wmt['translation']:
#     de = example['de']
#     en = example['en']
#     if len(de) <= max_len and len(en) <= max_len:
#         examples_valid_wmt.append(Example.fromlist([de, en], fields=[('de', SRC), ('en', TRG)]))

# valid_data_wmt = Dataset(examples_valid_wmt, fields=[('de', SRC), ('en', TRG)])


# examples_test_wmt = []
# for example in small_test_dataset_wmt['translation']:
#     de = example['de']
#     en = example['en']
#     if len(de) <= max_len and len(en) <= max_len:
#         examples_test_wmt.append(Example.fromlist([de, en], fields=[('de', SRC), ('en', TRG)]))

# test_data_wmt = Dataset(examples_test_wmt, fields=[('de', SRC), ('en', TRG)])
# print(len(train_data_wmt))
# print(len(valid_data_wmt))
# print(len(test_data_wmt))

# # %%
# SRC.build_vocab(train_data_wmt, min_freq=2)
# TRG.build_vocab(train_data_wmt, min_freq=2)

# print(f"SRC: {len(SRC.vocab)}")
# print(f"TRG: {len(TRG.vocab)}")

# # %%
# print(vars(train_data_wmt[50])['en'])
# print(vars(train_data_wmt[50])['de'])
# print(len(train_data_wmt[50].de))

# # %%
# import torch
# from torchtext.data import BucketIterator

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = 'cpu'

# BATCH_SIZE = 32


# train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
#     (train_data_wmt, valid_data_wmt, test_data_wmt),
#     batch_size=BATCH_SIZE,
#     # shuffle=True,
#     # repeat=True,
#     # sort=False,
#     sort_within_batch=True,
#     sort_key=lambda x: len(x.de),
#     device=device)



from torch.utils.data import Dataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WMT14Dataset(Dataset):
    def __init__(self,dataset,tokenize_de,tokenize_en,max_length,device,src_vocab=None,trg_vocab=None):
        self.device = device
        self.dataset = []
        self.max_length = max_length

        for example in dataset['translation']:
            de = example['de']
            en = example['en']
            if len(de) <= self.max_length and len(en) <= self.max_length:
                self.dataset.append([de,en])

        self.tokenize_de = tokenize_de
        self.tokenize_en = tokenize_en
        self.SRC = Field(tokenize=self.tokenize_de, init_token="<s>", eos_token="</s>", lower=True, batch_first=True)
        self.TRG = Field(tokenize=self.tokenize_en, init_token="<s>", eos_token="</s>", pad_token = "<pad>", lower=True, batch_first=True)

        if src_vocab is None:
            src_tokens = [self.tokenize_de(text[0]) for text in self.dataset]
            self.SRC.build_vocab(src_tokens)
        else:
            self.SRC.vocab = src_vocab
            
        if trg_vocab is None:
            trg_tokens = [self.tokenize_en(text[1]) for text in self.dataset]
            self.TRG.build_vocab(trg_tokens)
        else:
            self.TRG.vocab = trg_vocab

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]
        source_token = self.tokenize_de(text[0])
        source_token_indices = [self.SRC.vocab.stoi[token] for token in source_token]
        source_token_indices.insert(0,2)
        source_token_indices.append(3)
        #source_token_indices += [1] * (512 - len(source_token_indices))
        source_tensor = torch.tensor(source_token_indices, dtype=torch.long)

        
        target_token = self.tokenize_en(text[1])
        target_token_indices = [self.TRG.vocab.stoi[token] for token in target_token]
        target_token_indices.insert(0,2)
        target_token_indices.append(3)
        #target_token_indices += [1] * (512 - len(target_token_indices))
        target_tensor = torch.tensor(target_token_indices, dtype=torch.long)

        return {'de' : source_tensor, "en" : target_tensor}
        

def my_collate_fn(samples):
    collate_de = []
    collate_en = []
    max_len_de = max([len(sample['de']) for sample in samples])
    max_len_en = max([len(sample['en']) for sample in samples])
    
    for sample in samples:
        diff_de = max_len_de - len(sample['de'])
        if diff_de > 0 :
            one_pad_de = torch.ones(diff_de).to(dtype=torch.long)
            collate_de.append(torch.cat([sample['de'],one_pad_de]))
        else:
            collate_de.append(sample['de'])
        
        diff_en = max_len_en - len(sample['en'])
        if diff_en > 0 :
            one_pad_en = torch.ones(diff_en).to(dtype=torch.long)
            collate_en.append(torch.cat([sample['en'],one_pad_en]))
        else:
            collate_en.append(sample['en'])

    return {'de': torch.stack(collate_de), 'en': torch.stack(collate_en)}



# # dataset initialization
train_dataset = WMT14Dataset(small_train_dataset_wmt, tokenize_de, tokenize_en, 500, device)
print(len(train_dataset.SRC.vocab))
print(len(train_dataset.TRG.vocab))
val_dataset = WMT14Dataset(small_valid_dataset_wmt, tokenize_de, tokenize_en, 500, device, src_vocab=train_dataset.SRC.vocab, trg_vocab=train_dataset.TRG.vocab)
test_dataset = WMT14Dataset(small_test_dataset_wmt, tokenize_de, tokenize_en, 500, device, train_dataset.SRC.vocab, train_dataset.TRG.vocab)

sampler = DistributedSampler(dataset=train_dataset,shuffle=True)
# val_sampler = DistributedSampler(dataset=train_dataset,shuffle=True)
# test_sampler = DistributedSampler(dataset=train_dataset,shuffle=True)

print(len(train_dataset))

 # DataLoader initialization
train_dataloader = DataLoader(train_dataset, batch_size=128, collate_fn = my_collate_fn, sampler = sampler)
val_dataloader = DataLoader(val_dataset, batch_size=128, collate_fn = my_collate_fn, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=128, collate_fn = my_collate_fn, shuffle=False)


for i , batch in enumerate(train_dataloader):
    
    for j in batch['en']:
        print(len(j))
        for x in j:
            print(train_dataset.TRG.vocab.itos[x], end=" ")
        break
    
    print("----------------------------")
    
    for k in batch['de']:
        print(len(k))
        for y in k:
            print(train_dataset.SRC.vocab.itos[y], end=" ")
        break
    break


# %%

import torch
import torch.nn as nn
import torch.optim as optim
import math
import copy

# %%
# model architecture
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0
        
        self.d_model = d_model 
        self.num_heads = num_heads 
        self.d_k = d_model // num_heads 
        
        self.W_q = nn.Linear(d_model, d_model) 
        self.W_k = nn.Linear(d_model, d_model) 
        self.W_v = nn.Linear(d_model, d_model) 
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):

        attention_score = torch.matmul(Q, K.transpose(-2,-1)) / math.sqrt(self.d_k)
        masking_value = -1e+30 if attention_score.dtype == torch.float32 else -1e+4

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, value = masking_value)
        
        attn_probs = torch.softmax(attention_score, dim=-1)
        output = torch.matmul(self.dropout(attn_probs), V)
        
        return output
        
    def split_heads(self, x):

        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1,2)
        
    def combine_heads(self, x):

        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        
    def forward(self, Q, K, V, mask=None):

        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        
        return output

# %%
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))

# %%
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        pe.requires_grad = False
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        # div_term = torch.arange(0, d_model, 2).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        # pe[:, 0::2] = torch.sin(position / (10000 ** (div_term / d_model)))
        # pe[:, 1::2] = torch.cos(position / (10000 ** (div_term / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

# %%
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

# %%
class Encoder(nn.Module):
    
    def __init__(self, src_vocab_size, d_model, num_heads, d_ff, dropout, num_layers, max_seq_length):
        super(Encoder, self).__init__()
        self.encoderList = nn.ModuleList([(EncoderLayer(d_model, num_heads, d_ff, dropout))
                                          for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, mask):
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src) * math.sqrt(self.d_model)))
        enc_output = src_embedded
        
        for layer in self.encoderList:
            enc_output = layer(enc_output, mask)
            
        enc_output = self.norm(enc_output)

        return enc_output
        
    

# %%
import torch.nn.functional as F

class Decoder(nn.Module):
    
    def __init__(self, tgt_vocab_size, d_model, num_heads, d_ff, dropout, num_layers, max_seq_length):
        super(Decoder, self).__init__()
        self.decoderList = nn.ModuleList([(DecoderLayer(d_model, num_heads, d_ff, dropout)) for _ in range(num_layers)])
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, enc_output, src_mask, tgt_mask):
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt) * math.sqrt(self.d_model)))
        dec_output = tgt_embedded
        
        for dec_layer in self.decoderList:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        dec_output = self.norm(dec_output)
            
        output = self.fc_out(dec_output)
        
        return output
    
        
    
        

# %%
import numpy as np
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, rank):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, dropout, num_layers, max_seq_length)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, dropout, num_layers, max_seq_length)
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
    
    
    
    def generate_mask(self, src, tgt):
        src_mask = (src != 1).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 1).unsqueeze(1).unsqueeze(2)

        seq_length = tgt.size(1)
#         nopeak_mask2 = torch.tril(torch.ones((seq_length, seq_length))).bool().to(src.device
        nopeak_mask2 = torch.tril(torch.ones((seq_length, seq_length))).bool().to(rank)  
        tgt_mask = tgt_mask & nopeak_mask2
        
        return src_mask, tgt_mask
    

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, src_mask, tgt_mask)
        # output = F.log_softmax(output, dim=-1)
        
        return output

# %%
# hyperparameter
src_vocab_size = len(train_dataset.SRC.vocab)
tgt_vocab_size = len(train_dataset.TRG.vocab)
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 512
dropout = 0.1

# DP
# transformer = nn.DataParallel(Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device))

# DDP
transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout,rank).to(rank)
transformer = DDP(transformer, device_ids=[rank])


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(transformer):,} trainable parameters')

# %%
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

transformer.apply(initialize_weights)

# optimizer
class Opt:
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        if step is None:
            step = self._step
        return self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        
def get_std_opt(model):
    return Opt(model.module.d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# learning rate scheduler with optimizer
# class CustomLRScheduler:

#     def __init__(self, optimizer, model_dimension, num_of_warmup_steps):
#         self.optimizer = optimizer
#         self.model_size = model_dimension
#         self.num_of_warmup_steps = num_of_warmup_steps

#         self.current_step_number = 0

#     def step(self):
#         self.current_step_number += 1
#         current_learning_rate = self.get_current_learning_rate()

#         for p in self.optimizer.param_groups:
#             p['lr'] = current_learning_rate

#     def get_current_learning_rate(self):
#         # For readability purpose
#         step = self.current_step_number
#         warmup = self.num_of_warmup_steps

#         return self.model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

#     def zero_grad(self):
#         self.optimizer.zero_grad()

# # %%
# optimizer = optim.Adam(transformer.parameters(),betas=(0.9,0.98),eps=1e-9)
# scheduler = CustomLRScheduler(optimizer,d_model,4000)

# %%
import matplotlib.pyplot as plt
import numpy as np

criterion = nn.CrossEntropyLoss(ignore_index=1,label_smoothing=0.1)
optimizer = get_std_opt(transformer)

# plt.plot(np.arange(1, 20000), [optimizer._get_lr_scale(i) for i in range(1, 20000)])
# %%


# Training code(Accumulation with autocast, GradScaler)


# accumulation_steps = 2
# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# scaler = GradScaler()
# %%
# def train(model, iterator, scheduler, criterion, clip):
#     model.train()
#     epoch_loss = 0
#     total_batch = len(iterator)
#     for step, batch in enumerate(iterator):
#         src = batch['de'].to(device)
#         trg = batch['en'].to(device)
#         scheduler.zero_grad()
# #         with autocast():
#         output = model(src, trg[:,:-1])
#         output_dim = output.shape[-1]
#         output = output.contiguous().view(-1, output_dim)
#         trg = trg[:,1:].contiguous().view(-1)
#         loss = criterion(output, trg)
#         loss.backward()
# #             batch_loss += (loss / accumulation_steps)
# #         scaler.scale(loss).backward()
# #         if (step + 1) % accumulation_steps == 0:
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
# #             scaler.step(scheduler.optimizer)
# #             scaler.update()
#         scheduler.step()
#         scheduler.optimizer.step()
#         epoch_loss += loss.item()
#     return epoch_loss / total_batch
        
    
# Training code (No gradient accumulation)


# def train(rank, model, iterator, optimizer, criterion, clip, epoch):
#     sampler.set_epoch(epoch)
#     model.train() 
#     epoch_loss = 0

#     for i, batch in enumerate(iterator):
# #         src = batch['de'].to(device)
# #         trg = batch['en'].to(device)    
#         src = batch['de'].to(rank)
#         trg = batch['en'].to(rank)
#         optimizer.optimizer.zero_grad()
#         output = model(src, trg[:,:-1])
#         output_dim = output.shape[-1]
#         output = output.contiguous().view(-1, output_dim)
#         trg = trg[:,1:].contiguous().view(-1)
#         loss = criterion(output, trg)
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#         optimizer.step()
#         epoch_loss += loss.item()
        
#     num_samples = i + 1
#     return epoch_loss / num_samples



# Training code (gradient accumulation with no autocast, GradScaler)


accumulation_steps = 4

# %%
# def train(rank, model, iterator, optimizer, criterion, clip, epoch):
#     sampler.set_epoch(epoch)
#     model.train() 
#     epoch_loss = 0
#     batch_loss = 0
#     optimizer.optimizer.zero_grad()
#     steps = len(iterator)
#     acc_batch = steps // accumulation_steps
    
#     for i, batch in enumerate(iterator):
#         src = batch['de'].to(rank)
#         trg = batch['en'].to(rank)
        
        
#         output = model(src, trg[:,:-1])
#         output_dim = output.shape[-1]
#         output = output.contiguous().view(-1, output_dim)
#         trg = trg[:,1:].contiguous().view(-1)
#         loss = criterion(output, trg)
#         loss = loss / accumulation_steps
#         batch_loss += loss.item()
#         # epoch_loss +=  (loss / accumulation_steps)
#         loss.backward()
        
#         if (i + 1) % accumulation_steps == 0:
            
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
#             optimizer.step()
#             optimizer.optimizer.zero_grad()
#             epoch_loss += batch_loss
#             batch_loss = 0
#     return epoch_loss / acc_batch
        
        
        
# %%
import collections
import math

import torch
from torchtext.data.utils import ngrams_iterator


def _compute_ngram_counter(tokens, max_n):
    """Create a Counter with a count of unique n-grams in the tokens list

    Args:
        tokens: a list of tokens (typically a string split on whitespaces)
        max_n: the maximum order of n-gram wanted

    Outputs:
        output: a collections.Counter object with the unique n-grams and their
            associated count

    Examples:
        >>> from torchtext.data.metrics import _compute_ngram_counter
        >>> tokens = ['me', 'me', 'you']
        >>> _compute_ngram_counter(tokens, 2)
            Counter({('me',): 2,
             ('you',): 1,
             ('me', 'me'): 1,
             ('me', 'you'): 1})
    """
    assert max_n > 0
    ngrams_counter = collections.Counter(tuple(x.split(" ")) for x in ngrams_iterator(tokens, max_n))

    return ngrams_counter


def bleu_score(candidate_corpus, references_corpus, max_n=4, weights=[0.25] * 4):
    """Computes the BLEU score between a candidate translation corpus and a references
    translation corpus. Based on https://www.aclweb.org/anthology/P02-1040.pdf

    Args:
        candidate_corpus: an iterable of candidate translations. Each translation is an
            iterable of tokens
        references_corpus: an iterable of iterables of reference translations. Each
            translation is an iterable of tokens
        max_n: the maximum n-gram we want to use. E.g. if max_n=3, we will use unigrams,
            bigrams and trigrams
        weights: a list of weights used for each n-gram category (uniform by default)

    Examples:
        >>> from torchtext.data.metrics import bleu_score
        >>> candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
        >>> references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
        >>> bleu_score(candidate_corpus, references_corpus)
            0.8408964276313782
    """

    assert max_n == len(weights), 'Length of the "weights" list has be equal to max_n'
    assert len(candidate_corpus) == len(
        references_corpus
    ), "The length of candidate and reference corpus should be the same"

    clipped_counts = torch.zeros(max_n)
    total_counts = torch.zeros(max_n)
    weights = torch.tensor(weights)

    candidate_len = 0.0
    refs_len = 0.0

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        current_candidate_len = len(candidate)
        candidate_len += current_candidate_len

        # Get the length of the reference that's closest in length to the candidate
        refs_len_list = [float(len(ref)) for ref in refs]
        refs_len += min(refs_len_list, key=lambda x: abs(current_candidate_len - x))

        reference_counters = _compute_ngram_counter(refs[0], max_n)
        for ref in refs[1:]:
            reference_counters = reference_counters | _compute_ngram_counter(ref, max_n)

        candidate_counter = _compute_ngram_counter(candidate, max_n)

        clipped_counter = candidate_counter & reference_counters

        for ngram, count in clipped_counter.items():
            clipped_counts[len(ngram) - 1] += count

        for i in range(max_n):
            # The number of N-grams in a `candidate` of T tokens is `T - (N - 1)`
            total_counts[i] += max(current_candidate_len - i, 0)

    if min(clipped_counts) == 0:
        return 0.0
    else:
        pn = clipped_counts / total_counts
        log_pn = weights * torch.log(pn)
        score = torch.exp(sum(log_pn))

        bp = math.exp(min(1 - refs_len / candidate_len, 0))

        return bp * score.item()


# %%
# from torchtext.data.metrics import bleu_score
# from nltk.translate.bleu_score import sentence_bleu

def get_bleu_score(output, gt, vocab, specials, max_n=4):

    def itos(x):
        x = list(x.cpu().numpy())
        tokens = [vocab.itos[i] for i in x]
        tokens = list(filter(lambda x: x not in {"", " ", "."} and x not in specials, tokens))
        return tokens

    pred = [out.max(dim=1)[1] for out in output]
    # print("pred shape: ", (pred))
    pred_str = list(map(itos, pred))
    gt_str = list(map(lambda x: [itos(x)], gt))
    # print("pred_str: ", pred_str)
    # print("gt_str: ", gt_str)

    score = bleu_score(pred_str, gt_str, max_n=max_n) * 100
    return score

# %%
specials= {"<s>", "</s>", "<unk>", "<pad>"}

# %%
def evaluate(model, iterator, criterion):
    model.eval() 
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():

        for i, batch in enumerate(iterator):
            src = batch['de'].to(device)
            trg = batch['en'].to(device)
            
            trg_x = trg[:, :-1]
            trg_y = trg[:, 1:]

            output = model(src, trg_x)
            output_dim = output.shape[-1]

            y_hat = output.contiguous().view(-1, output_dim)
            y_gt = trg_y.contiguous().view(-1)
            loss = criterion(y_hat, y_gt)

            epoch_loss += loss.item()
            score = get_bleu_score(output, trg_y, train_dataset.TRG.vocab, specials)
            batch_bleu.append(score)
        num_samples = i + 1        
    
    return epoch_loss / num_samples, sum(batch_bleu) / len(batch_bleu)

# %%
import math
import time

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

# %%

N_EPOCHS = 10
CLIP = 1
best_valid_loss = 10 ** 9
patience_limit = 3
patience_check = 0

for epoch in range(N_EPOCHS):
    start_time = time.time()
    # train_loss = train(rank, transformer, train_dataloader, optimizer, criterion, CLIP, epoch)
    train(rank, transformer, train_dataloader, optimizer, criterion, CLIP, epoch)
    valid_loss, bleu = evaluate(transformer, val_dataloader, criterion)
    
    # if valid_loss > best_valid_loss: #early stopping
    #     patience_check += 1
    #     if patience_check >= patience_limit:
    #         break
    # else:
    #     best_valid_loss = valid_loss
    #     patience_check = 0

    end_time = time.time() 
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    # print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
    print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
    print(f'\tBLEU Score: {bleu:.3f}')



# %%
test_loss, bleu = evaluate(transformer, test_dataloader, criterion)

print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')
print(f'\tBLEU Score: {bleu:.3f}')


# %%
def translate_sentence(sentence, src_field, trg_field, model, device, max_len, logging):
    model.eval()
    
#     if isinstance(sentence, str):
#         tokens = [token.text.lower() for token in tokenize_de(sentence)]
#     else:
#         tokens = [token.lower() for token in sentence]

#     tokens = [src_field.init_token] + tokens + [src_field.eos_token]
#     if logging:
#         print(f"전체 소스 토큰: {tokens}")

#     src_indexes = [src_field.vocab.stoi[token] for token in tokens]
#     if logging:
#         print(f"소스 문장 인덱스: {src_indexes}")

    #src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src_tensor = torch.LongTensor(sentence).unsqueeze(0).to(device)

    

    src_mask = (src_tensor != 1).unsqueeze(1).unsqueeze(2)

    
    with torch.no_grad():
        enc_src = model.module.encoder(src_tensor, src_mask)

        

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        tgt_mask = (trg_tensor != 1).unsqueeze(1).unsqueeze(2)
        seq_length = trg_tensor.size(1)
        nopeak_mask2 = torch.tril(torch.ones((seq_length, seq_length))).bool().to(device)
        tgt_mask = tgt_mask & nopeak_mask2
        

        with torch.no_grad():
            output = model.module.decoder(trg_tensor, enc_src, src_mask, tgt_mask)

        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token) 

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:]

# %%
example_idx = 7

src = test_dataset[example_idx]['de']
trg = test_dataset[example_idx]['en']
#src = vars(test_iterator.dataset.examples[example_idx])['de']
#trg = vars(test_iterator.dataset.examples[example_idx])['en']

#print(f'소스 문장: {src}')
#print(f'타겟 문장: {trg}')

print(f'소스 문장: {"".join([test_dataset.SRC.vocab.itos[index] for index in src])}')
print(f'타겟 문장: {"".join([test_dataset.TRG.vocab.itos[index] for index in trg])}')

translation = translate_sentence(src, test_dataset.SRC, test_dataset.TRG, transformer, device, 512,logging=True)

print("모델 출력 결과:", "".join(translation))

# %%


# %%


# %%



