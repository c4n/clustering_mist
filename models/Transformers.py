"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
# from transformers import AutoModel, AutoTokenizer

class SCCLBert(nn.Module):
    def __init__(self, feature_extractor, cluster_centers=None, alpha=1.0, use_head = False):
        super(SCCLBert, self).__init__()
#         print(feature_extractor[0].tokenizer)  

        self.tokenizer = feature_extractor[0].tokenizer
        self.sentbert = feature_extractor[0].auto_model
        self.emb_size = self.sentbert.config.hidden_size
        
        self.use_head = use_head
        
        self.model = feature_extractor
        self.alpha = alpha
        
        # Instance-CL head
        if use_head == True :
            self.head = nn.Sequential(
                    nn.Linear(self.emb_size, self.emb_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.emb_size, self.emb_size))

        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)
        
#         self.register_parameter('cluster_centers', self.cluster_centers)

        
        print("initial_cluster_centers = ",initial_cluster_centers.shape)
        
    def forward(self, inputs, args = None, batch = None):
#         print('===== In SCCLBert forward =====')
        inputs.update({'args': args, 'batch': batch})
#         print('input keys:', inputs.keys())
        features = self.model(inputs)
#         features = self.model(inputs)
        
        return features

    def get_embeddings(self, features, pooling="mean"):
        bert_output = self.sentbert.forward(**features)
        attention_mask = features['attention_mask'].unsqueeze(-1)
        all_output = bert_output[0]
        mean_output = torch.sum(all_output*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def get_cluster_prob(self, embeddings):
        
        norm_squared = torch.sum((embeddings.unsqueeze(1) - self.cluster_centers) ** 2, 2)
        numerator = 1.0 / (1.0 + (norm_squared / self.alpha))
        power = float(self.alpha + 1) / 2
        numerator = numerator ** power
        return numerator / torch.sum(numerator, dim=1, keepdim=True)

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2

    







