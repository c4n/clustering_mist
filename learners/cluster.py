"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .cluster_utils import target_distribution
from .contrastive_utils import SupConLoss
from .criterion import KCL
from sentence_transformers_local import models, losses
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer

class ClusterLearner(nn.Module):
    def __init__(self, model, feature_extractor, optimizer, temperature, base_temperature, contrastive_local_scale, contrastive_global_scale, clustering_scale, use_head = False, use_normalize = False):
        
        super(ClusterLearner, self).__init__()
        
        self.model = model
        self.feature_extractor = feature_extractor
        self.optimizer = optimizer
        
        self.use_head = use_head
        self.use_normalize = use_normalize
        self.contrastive_local_scale = contrastive_local_scale
        self.contrastive_global_scale = contrastive_global_scale
        self.clustering_scale = clustering_scale
        #self.avg_sentence_length = 0.0
        
        # Loss_1 & Loss_2 = Local_MI_Maximization Loss & Global_MI_Maximization
        self.contrast_loss =  losses.MutualInformationLoss(model=model,
                              sentence_embedding_dimension=feature_extractor.get_sentence_embedding_dimension())
        
       # Loss_3 = Clustering Loss
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.kcl = KCL()

    def forward(self, inputs, args, batch, use_perturbation=False):
        
        # get sentence representations of "original texts" for clustering task
        # if args.is_augmented == False :
        features_0 = self.model(inputs[0], args, batch) 
        sentence_rep_0 = features_0['sentence_embedding']
#         print('batch:', torch.tensor(batch['index'], dtype=torch.int).size())
        
        
        # get features of augmented texts
        # if args.is_augmented == True :
        features_1 = self.model(inputs[1], args, batch) 
        
        # get features of augmented texts
#         args.is_augmented = True
#         features_2 = self.model(inputs[2], args, batch) 
          
        # Local_MI_Maximization Loss
        contrastive_local_loss,avg_sentence_length = self.contrast_loss(args, batch, features_0, features_1, self.use_head, self.use_normalize, objective = "local")
        
       # Global_MI_Maximization Loss
        contrastive_global_loss,_ =  self.contrast_loss(args, batch, features_0, features_1, self.use_head, self.use_normalize, objective = "global")
        
        loss = 0.0
        
#         # maximize : BERT and CNN
#         contrastive_loss_bert =  self.contrast_loss(features, self.use_head, self.use_normalize, token = 'bert')
#         contrastive_loss_cnn =  self.contrast_loss(features, self.use_head, self.use_normalize, token = 'cnn')
#         contrastive_loss = (contrastive_loss_bert + contrastive_loss_cnn)/2
        
        # Representation Losses : local + global
        
        # Adaptive Weight : we use ***round function*** in Eq.3
            
#         #contrastive_local_scale = avg_sentence_length*(10**-1) - 1 # method_1
#         #contrastive_local_scale = torch.ceil(torch.tensor(avg_sentence_length*(10**-1) - 1)) # method_2 : ceiling
#         #contrastive_local_scale = torch.floor(torch.tensor(avg_sentence_length*(10**-1) - 1)) # method_3 : floor
        #self.contrastive_local_scale = torch.round(torch.tensor(avg_sentence_length)*(10**-1))-1 # method_4 : round
        self.contrastive_local_scale = torch.round(torch.tensor(avg_sentence_length)*(10**-1)-1)
#        contrastive_local_scale = torch.round(torch.tensor(avg_sentence_length))*(10**-1)-1 # method_5 : inner_round
#         ##contrastive_local_scale = torch.round(torch.tensor(avg_sentence_length)*(10**-1))-1 # method_6 : outer_round

        if self.contrastive_local_scale < 0 or self.contrastive_local_scale == 0.0 :
            self.contrastive_local_scale = 0.0*(10**-3)
            #contrastive_local_loss *= self.contrastive_local_scale*(10**-3) 
            #print('contrastive_local_scale_1 :', self.contrastive_local_scale)
            contrastive_local_loss *= self.contrastive_local_scale
            loss = contrastive_local_loss
        else:
            self.contrastive_local_scale = self.contrastive_local_scale*(10**-3)  
            #print('contrastive_local_scale_2 :', self.contrastive_local_scale)
            contrastive_local_loss *= self.contrastive_local_scale
            loss = contrastive_local_loss

        #print('loss_local=',loss)
        
#         if self.contrastive_local_scale == 0.0:
#             loss = 0.0
#         else:
#             contrastive_local_loss *= self.contrastive_local_scale
#             loss = contrastive_local_loss
       
        
    
#         contrastive_local_loss *= self.contrastive_local_scale
#         loss = contrastive_local_loss 
        
        contrastive_global_scale = (10**-2) - self.contrastive_local_scale
        contrastive_global_loss *= self.contrastive_global_scale
        ##print('contrastive_local_scale :', self.contrastive_local_scale*(10**-3))
        #print('contrastive_global_scale :', contrastive_global_scale)
        loss += contrastive_global_loss
            
            
        # Clustering Loss
        output = self.model.get_cluster_prob(sentence_rep_0)
        target = target_distribution(output).detach()
        cluster_loss = self.cluster_loss((output+1e-08).log(),target)/output.shape[0]
        cluster_loss *=  self.clustering_scale
        loss += cluster_loss

        # consistency loss (this loss is used in the experiments of our NAACL paper, we included it here just in case it might be helpful for your specific applications)
        local_consloss_val = 0
        #if use_perturbation:
        #local_consloss = self.model.local_consistency(embd0, embd1, embd2, self.kcl)
        # loss += local_consloss
        # local_consloss_val = local_consloss.item()

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"contrastive_local_loss":contrastive_local_loss.detach(), "contrastive_global_loss":contrastive_global_loss.detach(),"clustering_loss":cluster_loss.detach(), "local_consistency_loss":local_consloss_val}
