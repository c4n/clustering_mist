import torch
from torch import Tensor
from torch import nn
from typing import Union, Tuple, List, Iterable, Dict
import os
import json
import pickle
import numpy as np


class Pooling(nn.Module):
    """Performs pooling (max or mean) on the token embeddings.

    Using pooling, it generates from a variable sized sentence a fixed sized sentence embedding. This layer also allows to use the CLS token if it is returned by the underlying word embedding model.
    You can concatenate multiple poolings together.

    :param word_embedding_dimension: Dimensions for the word embeddings
    :param pooling_mode_cls_token: Use the first token (CLS token) as text representations
    :param pooling_mode_max_tokens: Use max in each dimension over all tokens.
    :param pooling_mode_mean_tokens: Perform mean-pooling
    :param pooling_mode_mean_sqrt_len_tokens: Perform mean-pooling, but devide by sqrt(input_length).
    """
    def __init__(self,
                 word_embedding_dimension: int,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 pooling_mode_weighted_tokens: bool = False,
                 ):
        super(Pooling, self).__init__()

        self.config_keys = ['word_embedding_dimension',  'pooling_mode_cls_token', 'pooling_mode_mean_tokens', 'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens', 'pooling_mode_weighted_tokens']

        self.word_embedding_dimension = word_embedding_dimension
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens
        self.pooling_mode_weighted_tokens = pooling_mode_weighted_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens, pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * word_embedding_dimension)

    def forward(self, features: Dict[str, Tensor]):
        
#         print('===== In Pooling forward =====')
        
        token_embeddings = features['token_embeddings']
        #for experiment : token = BERT
        #token_embeddings = features['token_embeddings_cnn']
        
        cls_token = features['cls_token_embeddings']
        attention_mask = features['attention_mask']
        sentence_lengths = torch.sum(attention_mask, dim=1)
        features.update({'sentence_lengths': sentence_lengths})

        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = torch.max(token_embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)

            #If tokens are weighted (by WordWeights layer), feature 'token_weights_sum' will be present
            if 'token_weights_sum' in features:
                sum_mask = features['token_weights_sum'].unsqueeze(-1).expand(sum_embeddings.size())
            else:
                sum_mask = input_mask_expanded.sum(1)

            sum_mask = torch.clamp(sum_mask, min=1e-9)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / torch.sqrt(sum_mask))
                
        if self.pooling_mode_weighted_tokens:
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#             print(features.keys())
            args = features['args']
            batch = features['batch']
            
            weight = self.load_weight(args.dataname, is_augmented = args.is_augmented)
            
            #CLEAR PADDED EMBEDDINGS
            embeddings_expanded = token_embeddings * input_mask_expanded

            index = batch['index']              
            
            #Calculated each sentence
            for idx, length in enumerate(sentence_lengths):
                              
                tokens_weight = weight[index[idx]]                
                
                if args.normalize_method == 'inverse_prob':
                    inv = 1.0 / (1.0 - np.array(tokens_weight))
                    sum_inv = sum(inv)
                    normalized_tokens_weight = inv * length.item() / sum_inv
                    
                elif args.normalize_method == 'none':
                    sum_weight = sum(np.array(tokens_weight))
                    normalized_tokens_weight = tokens_weight * length.item() / sum_weight
                
                sum_normalized_tokens_weight = sum(normalized_tokens_weight)

                for l in range(length):
                    embeddings_expanded[idx][l] *= normalized_tokens_weight[l]           

                sum_embeddings = torch.sum(embeddings_expanded[idx], dim = 0)
                output_vectors.append(sum_embeddings / sum_normalized_tokens_weight)
        
            output_vector = torch.vstack(output_vectors)
            features.update({'sentence_embedding': output_vector})
            return features
        
        output_vector = torch.cat(output_vectors, 1)
        features.update({'sentence_embedding': output_vector})
        return features

    def get_sentence_embedding_dimension(self):
        return self.pooling_output_dimension

    def get_config_dict(self):
        return {key: self.__dict__[key] for key in self.config_keys}

    def save(self, output_path):
        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(self.get_config_dict(), fOut, indent=2)

    @staticmethod
    def load(input_path):
        with open(os.path.join(input_path, 'config.json')) as fIn:
            config = json.load(fIn)

        return Pooling(**config)    

    
    def load_weight(self, dataname, is_augmented):

        weight_path = 'datasets/augmented/contextual_20_2col_bert/weight/'

        if dataname == 'agnewsdataraw-8000':
            weight_name = '_weight_agnews_'

        elif dataname == 'search_snippets':
            weight_name = '_weight_search_snippets_'

        elif dataname == 'stackoverflow':
            weight_name = '_weight_stackoverflow_'

        elif dataname == 'biomedical':
            weight_name = '_weight_biomedical_'

        elif dataname == 'tweet_remap_label':
            weight_name = '_weight_tweet_'

        elif dataname == 'TS':
            weight_name = '_weight_TS_'

        elif dataname == 'T':
            weight_name = '_weight_T_'

        elif dataname == 'S':
            weight_name = '_weight_S_'

        if not is_augmented:
            text = 'text0'
        else:
            text = 'text1'

        with open(weight_path + weight_name + text+ '.pkl', 'rb') as f:
            weight_file = pickle.load(f)

        return weight_file

