import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
import torch.nn.functional as F
from ..SentenceTransformerSequential import SentenceTransformerSequential
import torch
import torch.nn as nn
import numpy as np
import logging
import math
import pickle

class MutualInformationLoss(nn.Module):
    def __init__(self,
                 model,
                 sentence_embedding_dimension: int):
        super(MutualInformationLoss, self).__init__()
        self.model = model


        
#     def forward(self, features_0, features_1, use_head = False, use_normalize = False, objective = "local", note = "only"):       
    def forward(self, args, batch, features_0, features_1, use_head = False, use_normalize = False, objective = "local", note = " "):  
                
        self.weight_0 = self.load_weight(args.dataname, is_augmented = False)
        self.weight_1 = self.load_weight(args.dataname, is_augmented = True)
#         self.weight_2 = self.load_weight(args.dataname, is_augmented = True)
        
        avg_sentence_length = 0.0
#         local_rep_0  = torch.tensor()
#         local_rep_1 = torch.tensor()
        
#         global_rep_0 = torch.tensor()
#         global_rep_1 = torch.tensor()
        
        if objective == "local":
            
            tok_rep_0 = features_0['token_embeddings']
            tok_rep_1 = features_1['token_embeddings']
#             tok_rep_2 = features_2['token_embeddings']
             
            sentence_0_lengths = torch.clamp(features_0['sentence_lengths'], min=1).data.cpu().numpy() 
            #print("sentence_length shape:",sentence_0_lengths.shape)
            #print('sentence_length_0 mean:', sentence_0_lengths.mean())
            
            sentence_1_lengths = torch.clamp(features_1['sentence_lengths'], min=1).data.cpu().numpy() 
            #print('sentence_length_1 mean:', sentence_1_lengths.mean())
            
#             sentence_2_lengths = torch.clamp(features_2['sentence_lengths'], min=1).data.cpu().numpy() 
            #print('sentence_length_1 mean:', sentence_1_lengths.mean())
            
            #sentence_0_1_lengths = torch.cat((sentence_0_lengths, sentence_1_lengths), dim=0) 
            avg_sentence_length = (sentence_0_lengths.mean() + sentence_1_lengths.mean())/2
#             avg_sentence_length = (sentence_1_lengths.mean() + sentence_2_lengths.mean())/2
            #print('sentence_length_0_1 mean:',avg_sentence_length)
          
            tok_rep_0 = [tok_rep_0[i][:sentence_0_lengths[i]] for i in range(len(sentence_0_lengths))]
            tok_rep_1 = [tok_rep_1[i][:sentence_1_lengths[i]] for i in range(len(sentence_1_lengths))]
#             tok_rep_2 = [tok_rep_2[i][:sentence_2_lengths[i]] for i in range(len(sentence_2_lengths))]
            
            local_rep_0 = torch.cat(tok_rep_0, dim=0)
            local_rep_1 = torch.cat(tok_rep_1, dim=0)
#             local_rep_2 = torch.cat(tok_rep_2, dim=0)

            global_rep_0 = features_0['sentence_embedding']
            global_rep_1 = features_1['sentence_embedding']
#             global_rep_2 = features_2['sentence_embedding']
       
        elif objective == "global":
            
            global_rep_0 = features_0['sentence_embedding']
            global_rep_1 = features_1['sentence_embedding']
#             global_rep_2 = features_2['sentence_embedding']
                    
        #use_head = True
        if use_head == True :
#             pass
#             local_rep = self.model.head(local_rep)
#             global_rep = self.model.head(global_rep)
#             local_rep_0 = self.model.head(local_rep_0)
#             local_rep_1 = self.model.head(local_rep_1)
#             print('Before head')
#             print('global_rep_0.shape = ', global_rep_0.shape)
#             print('global_rep_1.shape = ', global_rep_1.shape)
            
            global_rep_0 = self.model.head(global_rep_0)
            global_rep_1 = self.model.head(global_rep_1)
            
#             print('=======================')
#             print('After head')
#             print('global_rep_0.shape = ', global_rep_0.shape)
#             print('global_rep_1.shape = ', global_rep_1.shape)      

        
        if use_normalize == True :
            local_rep = F.normalize(local_rep, dim=1)
            global_rep = F.normalize(global_rep, dim=1)

        # creat pos_mask, neg_pask
        if objective == "local":
            
            pos_mask_0, neg_mask_0 = self.create_local_masks(sentence_0_lengths)
            
   
            if args.weighted_local:
#                 print('------USED weighted_norm-----')
                weighted_pos_mask_0 = self.weighted_norm(
                    sentence_0_lengths, batch, pos_mask_0, args.normalize_method, is_augmented = False)
            else:
#                 print('------NOT USED weighted_norm-----')
                pass

            if note == "only" :
                pos_mask = pos_mask_0 
                neg_mask = neg_mask_0
                
            else :
                pos_mask_1, neg_mask_1 = self.create_local_masks(sentence_1_lengths)
#                 pos_mask_2, neg_mask_2 = self.create_local_masks(sentence_2_lengths)
                
                # method 1 + method 3 + method 4
                pos_mask = torch.cat((pos_mask_0, pos_mask_1), dim=0) 
                neg_mask = torch.cat((neg_mask_0, neg_mask_1), dim=0)
                
                # only method 5 : only original texts,discard augmented
#                 pos_mask = pos_mask_0 
#                 neg_mask = neg_mask_0
                
                # only method 3 : Local DIM
#                 pos_mask_aug = torch.zeros((np.sum(sentence_1_lengths), len(sentence_1_lengths))).cuda()
#                 pos_mask_ori = torch.zeros((np.sum(sentence_0_lengths), len(sentence_0_lengths))).cuda()
#                 pos_mask_t = torch.cat((pos_mask_aug, pos_mask_ori), dim=0) 
#                 pos_mask = torch.cat((pos_mask, pos_mask_t), dim=0) 
                
#                 neg_mask_aug = torch.ones((np.sum(sentence_1_lengths), len(sentence_1_lengths))).cuda()
#                 neg_mask_ori = torch.ones((np.sum(sentence_0_lengths), len(sentence_0_lengths))).cuda()
#                 neg_mask_t = torch.cat((neg_mask_aug, neg_mask_ori), dim=0)
#                 neg_mask = torch.cat((neg_mask, neg_mask_t), dim=0)
                
                # only method 4 : Local DIM + AMDIM
#                 pos_mask_t = torch.cat((pos_mask_1, pos_mask_0), dim=0) 
#                 pos_mask = torch.cat((pos_mask, pos_mask_t), dim=0) 
#                 neg_mask_t = torch.cat((neg_mask_1, neg_mask_0), dim=0)
#                 neg_mask = torch.cat((neg_mask, neg_mask_t), dim=0)
                
                
              
                if args.weighted_local:
                    weighted_pos_mask_1 = self.weighted_norm(
                        sentence_1_lengths, batch, pos_mask_1, args.normalize_method, is_augmented = True)
                    weighted_pos_mask = torch.cat((weighted_pos_mask_0, weighted_pos_mask_1), dim=0) 
                else :
                    weighted_pos_mask = pos_mask.clone().detach()
            
        elif objective == "global":
            pos_mask, neg_mask = self.create_global_masks(global_rep_0.size(0))

#         mode='fd'
        measure='JSD'
#         measure = 'DV'

        if objective == "local":
        
            res_0 = torch.mm(local_rep_0, global_rep_0.t())
            res_1 = torch.mm(local_rep_1, global_rep_1.t())
            
            res = torch.cat((res_0, res_1), dim=0)
            
            # only method 5 : only original texts,discard augmented
            #res = res_0
            
            # only method 3 : Local DIM
            # only method 4 : Local DIM + AMDIM
#             res_2 = torch.mm(local_rep_1, global_rep_0.t())
#             res_3 = torch.mm(local_rep_0, global_rep_1.t())
#             res_t = torch.cat((res_2, res_3), dim=0)
#             res = torch.cat((res, res_t), dim=0)
            
            if note == "only" :
                res = res_0
                
            if args.weighted_local:
                local_global_loss = self.local_global_loss_(res, weighted_pos_mask, pos_mask, neg_mask, measure, use_normalize,objective)
            else:
                local_global_loss = self.local_global_loss_(res, pos_mask, pos_mask, neg_mask, measure, use_normalize,                             objective)
                 # เพิ่ม objective มาตอนที่ทำ only method 3 : Local DIM
            
        elif objective == "global":
            
            weighted_pos_mask = pos_mask.clone().detach()
            
            res = torch.mm(global_rep_0, global_rep_1.t())
#             res = torch.mm(global_rep_1, global_rep_2.t())

            local_global_loss = self.local_global_loss_(res, pos_mask, pos_mask, neg_mask, measure, use_normalize, objective)
             # เพิ่ม objective มาตอนที่ทำ only method 3 : Local DIM
        
        #avg_sentence_length = sentence_0_lengths,sentence_1_lengths
        
#         print('weight_pos_mask:', weighted_pos_mask)
#         print('pos_mask:', pos_mask)
        
        return local_global_loss, avg_sentence_length

    def create_local_masks(self, lens_a):
        pos_mask = torch.zeros((np.sum(lens_a), len(lens_a))).cuda()
        neg_mask = torch.ones((np.sum(lens_a), len(lens_a))).cuda()
        temp = 0
        for idx in range(len(lens_a)):
            for j in range(temp, lens_a[idx] + temp):
                pos_mask[j][idx] = 1.
                neg_mask[j][idx] = 0.
            temp += lens_a[idx]

        return pos_mask, neg_mask

    def create_global_masks(self, batch_size):
        pos_mask = torch.zeros((batch_size, batch_size)).cuda()
        neg_mask = torch.ones((batch_size, batch_size)).cuda()
        for idx in range(batch_size):
            pos_mask[idx][idx] = 1.
            neg_mask[idx][idx] = 0.

        return pos_mask, neg_mask    
    

    def local_global_loss_(self, res, weighted_pos_mask, pos_mask, neg_mask, measure, use_normalize, objective):
        '''
        Args:
            l: Local feature map.
            g: Global features.
            measure: Type of f-divergence. For use with mode `fd`
            mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
        Returns:
            torch.Tensor: Loss.
        '''
#         print('l_enc.shape', l_enc.shape)
#         print('g_enc.t()', g_enc.t().shape)

#        res = torch.mm(l_enc, g_enc.t())
    
#        res = F.normalize(res)
#         print('res.shape', res.shape)

#        print('res =', res)

        # print(l_enc.size(), res.size(), pos_mask.size())
    
      
        num_nodes = pos_mask.size(0)        
        num_graphs = pos_mask.size(1)
        E_pos = self.get_positive_expectation(res * weighted_pos_mask, neg_mask, measure, use_normalize, average=False).sum()
        
        # method 1 + method 4 + method 5
        E_pos = E_pos / num_nodes 
        # only method 3 : Local DIM
#         if objective == "local":
#             E_pos = E_pos / (num_nodes/2)
#         elif objective == "global":
#             E_pos = E_pos / num_nodes
            
        E_neg = self.get_negative_expectation(res * neg_mask, pos_mask, measure, use_normalize, average=False).sum()
        
        # method 1 + method 4 + method 5
        E_neg = E_neg / (num_nodes * (num_graphs - 1))
        # only method 3 : Local DIM
#         if objective == "local":
#             E_neg = E_neg / ((num_nodes * num_graphs) - (num_nodes/2))
#         elif objective == "global":
#             E_neg = E_neg / (num_nodes * (num_graphs - 1))
        
#         print('E_pos', E_pos)
#         print('E_neg', E_neg)

        return E_neg - E_pos
        #return  E_pos-E_neg


    def log_sum_exp(self, x, axis=None):
        """Log sum exp function

        Args:
            x: Input.
            axis: Axis over which to perform sum.

        Returns:
            torch.Tensor: log sum exp

        """
        x_max = torch.max(x, axis)[0]
        y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
        return y



    def raise_measure_error(self, measure):
        supported_measures = ['GAN', 'JSD', 'X2', 'KL', 'RKL', 'DV', 'H2', 'W1']
        raise NotImplementedError(
            'Measure `{}` not supported. Supported: {}'.format(measure,
                                                               supported_measures))


    def get_positive_expectation(self, p_samples, neg_mask, measure, use_normalize, average=True):
        """Computes the positive part of a divergence / difference.

        Args:
            p_samples: Positive samples.
            measure: Measure to compute for.
            average: Average the result over samples.

        Returns:
            torch.Tensor

        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Ep = - F.softplus(-p_samples)
        elif measure == 'JSD':
            if use_normalize == True:
                Ep = 0 - F.softplus(- p_samples) 
                #print("Ep =",Ep)
            else:
                #Ep = log_2 - F.softplus(- p_samples)
                Ep = 0 - F.softplus(- p_samples) + (log_2*neg_mask)
                
        elif measure == 'X2':
            Ep = p_samples ** 2
        elif measure == 'KL':
            Ep = p_samples + 1.
        elif measure == 'RKL':
            Ep = -torch.exp(-p_samples)
        elif measure == 'DV':
            Ep = p_samples
        elif measure == 'H2':
            Ep = 1. - torch.exp(-p_samples)
        elif measure == 'W1':
            Ep = p_samples
        else:
            raise_measure_error(measure)

        if average:
            return Ep.mean()
        else:
            return Ep

    def weighted_norm(self, length, batch, pos_mask, normalize_method, is_augmented = True):
        
        #compute raw_weight
        index = batch['index']
        
#         print(batch['text0'])
        temp, temp_2, temp_index = 0, 0, 0
        
        if not is_augmented:
            weight = self.weight_0
        else:
            weight = self.weight_1
            
        old_pos_mask = pos_mask.clone().detach()
        new_pos_mask = pos_mask.clone().detach()
        
        # Raw weights
        for idx in range(len(length)):
#             print('index:', index[idx].item())
            temp_index = 0
            for j in range(temp, length[idx] + temp):
#                 print('j:',j , 'idx:', idx, 'weight:', weight[index[idx]][temp_index] * 1.0)
#                 print('temp:', temp, 'length[idx]:', length[idx], 'temp_index:', temp_index, 'index:', index[idx].item())
                new_pos_mask[j][idx] = weight[index[idx]][temp_index] * 1.0
                temp_index += 1
            temp += length[idx]
            
#         print('pos_mask before inverse:', new_pos_mask[0:5])
        
        #Inverse probability weighting
        if normalize_method == 'inverse_prob':
            new_pos_mask = 1.0 / (1.0 - new_pos_mask)
            new_pos_mask *= old_pos_mask
            
        elif normalize_method == 'none':   
            pass
#         print('pos_mask before nomalize:', new_pos_mask[0:5])
        
        #Normalize weights
        sum_weights_in_sentence = new_pos_mask.sum(dim = 0)
#         print('sum_weights_in_sentence:',sum_weights_in_sentence)

        for idx_2 in range(len(length)):
            for k in range(temp_2, length[idx_2] + temp_2):
                new_pos_mask[k][idx_2] = (new_pos_mask[k][idx_2] * length[idx_2]) / sum_weights_in_sentence[idx_2]
            temp_2 += length[idx_2]
            
#         print('pos_mask after nomalize:', new_pos_mask[0:5])
        
        return new_pos_mask
        
    
    def get_negative_expectation(self, q_samples, pos_mask, measure, use_normalize, average=True):
        """Computes the negative part of a divergence / difference.

        Args:
            q_samples: Negative samples.
            measure: Measure to compute for.
            average: Average the result over samples.

        Returns:
            torch.Tensor

        """
        log_2 = math.log(2.)

        if measure == 'GAN':
            Eq = F.softplus(-q_samples) + q_samples
        elif measure == 'JSD':
            if use_normalize == True:
                Eq = F.softplus(-q_samples) + q_samples - log_2*pos_mask
                #print("Eq =",Eq)
            else:
                #Eq = F.softplus(-q_samples) + q_samples - log_2
                Eq = F.softplus(-q_samples) + q_samples - (log_2*pos_mask)
        elif measure == 'X2':
            Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
        elif measure == 'KL':
            Eq = torch.exp(q_samples)
        elif measure == 'RKL':
            Eq = q_samples - 1.
        elif measure == 'DV':
            Eq = self.log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
        elif measure == 'H2':
            Eq = torch.exp(q_samples) - 1.
        elif measure == 'W1':
            Eq = q_samples
        else:
            self.raise_measure_error(measure)

        if average:
            return Eq.mean()
        else:
            return Eq
        
    def load_weight(self, dataname, is_augmented):
        
        #weight_path = 'datasets/augmented/contextual_20_2col_bert/weight/'
        weight_path = 'datasets/augmented/contextual_20/weight/'
        
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
    

    