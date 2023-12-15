import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import logging
import gzip
from tqdm import tqdm
import numpy as np
import os
import json
from ..util import import_from_string, fullname, http_get
from .tokenizer import WordTokenizer, WhitespaceTokenizer


class CNN(nn.Module):
    """CNN-layer with multiple kernel-sizes over the word embeddings"""
    #[1, 3, 5]
    def __init__(self, in_word_embedding_dimension: int, use_cnn: str = 'cnn_1', out_channels: int = 256, kernel_sizes: List[int] = [3]):
        nn.Module.__init__(self)
        self.config_keys = ['in_word_embedding_dimension', 'out_channels', 'kernel_sizes']
        self.in_word_embedding_dimension = in_word_embedding_dimension
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.use_cnn = use_cnn

        self.embeddings_dimension = out_channels*len(kernel_sizes)
        self.convs = nn.ModuleList()

        in_channels = in_word_embedding_dimension
        for kernel_size in kernel_sizes:
            padding_size = int((kernel_size - 1) / 2)
            conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             padding=padding_size)
            self.convs.append(conv)

    def forward(self, features):
        token_embeddings = features['token_embeddings']
#         print('token_embeddings.shape', token_embeddings.shape)
        token_embeddings = token_embeddings.transpose(1, -1)
#         print('token_embeddings.shape after transpose', token_embeddings.shape)
        vectors = [conv(token_embeddings) for conv in self.convs]
#         print('vectors[0].shape', vectors[0].shape)
        #print('vector =',vectors)
        if self.use_cnn in ['cnn_1','cnn_3','cnn_5','cnn_7']:
            out = torch.stack(vectors).squeeze().transpose(1, -1)
        elif self.use_cnn == 'cnn_avg':
#             vectors = torch.stack(vectors)
            if len(vectors) == 2:
                cat = torch.cat([vectors[0].unsqueeze(1) , vectors[1].unsqueeze(1)], dim =1)
            elif len(vectors) == 3:    
                cat = torch.cat([vectors[0].unsqueeze(1) , vectors[1].unsqueeze(1), vectors[2].unsqueeze(1)], dim =1)
            out = torch.mean(cat, dim =1).transpose(1, -1)
        else:
            out = torch.cat(vectors, 1).transpose(1, -1)
            
            
        features.update({'token_embeddings': out})
        
        #for experiment : token = BERT
        #features.update({'token_embeddings_cnn': out})
        
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def tokenize(self, text: str) -> List[int]:
        raise NotImplementedError()

    def save(self, output_path: str):
        with open(os.path.join(output_path, 'cnn_config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

        torch.save(self.state_dict(), os.path.join(output_path, 'pytorch_model.bin'))

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, 'cnn_config.json'), 'r') as fIn:
            config = json.load(fIn)

        weights = torch.load(os.path.join(input_path, 'pytorch_model.bin'))
        model = CNN(**config)
        model.load_state_dict(weights)
        return model

