"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset, random_split
import math
import torch

class TextClustering(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}
    

class AugmentPairSamples(Dataset):
    def __init__(self, train_x_0, train_y, train_x_1 = None, train_x_2 = None):
        
        if train_x_1 is not None:
#             assert len(train_y) == len(train_x_0) == len(train_x_1) == len(train_x_2)
            assert len(train_y) == len(train_x_0) == len(train_x_1)
        else:
            assert len(train_y) == len(train_x_0)
            
        self.train_x_0 = train_x_0
        self.train_x_1 = train_x_1
#         self.train_x_2 = train_x_2
        self.train_y = train_y                
        
        
    def __len__(self):
        return len(self.train_y)

    def __getitem__(self, idx):
        if self.train_x_1 is not None:
            return {'text0': self.train_x_0[idx], 'text1': self.train_x_1[idx],'label': self.train_y[idx], 'index': idx}
#             return {'text0': self.train_x_0[idx], 'text1': self.train_x_1[idx], 'text2': self.train_x_2[idx],
#                     'label': self.train_y[idx], 'index': idx}
        else:
            return {'text0': self.train_x_0[idx], 'label': self.train_y[idx], 'index': idx}

def augment_loader(args):
    
##     W/o augmentation : read csv and set the name of headers
#     train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), names=["label", "text", "_"])



    # For Paraphrase
#     train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep='\t')
    
    # For Contextual
    train_data = pd.read_csv(os.path.join(args.data_path, args.dataname))
    
##     W/o augmentation
#     train_data['text'] = train_data['text'].str.strip()
    
##     W/o augmentation : has only original text and label 
#     train_text = train_data['text'].fillna('.').values
#     train_label = train_data['label'].astype(int).values

##    W/ augmentation : original text and 1 augmented data
    train_text_0 = train_data['text0'].fillna('.').values
    train_text_1 = train_data['text1'].fillna('.').values
#     train_text_2 = train_data['text2'].fillna('.').values
    train_label = train_data['label'].astype(int).values

##    W/o augmentation 
#     train_dataset = AugmentPairSamples(train_text, train_label)
    train_dataset = AugmentPairSamples(train_text_0, train_label, train_text_1)
#     train_dataset = AugmentPairSamples(train_text_0, train_label, train_text_1, train_text_2)
    
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    return train_loader

def augment_loader_split(args):
    
    ##     W/o augmentation : read csv and set the name of headers
#     train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), names=["label", "text", "_"])



    # For Paraphrase
#     train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep='\t')
    
    # For Contextual
    train_data = pd.read_csv(os.path.join(args.data_path, args.dataname))
    
##     W/o augmentation
#     train_data['text'] = train_data['text'].str.strip()
    
##     W/o augmentation : has only original text and label 
#     train_text = train_data['text'].fillna('.').values
#     train_label = train_data['label'].astype(int).values

##    W/ augmentation : original text and 1 augmented data
    train_text_0 = train_data['text0'].fillna('.').values
    train_text_1 = train_data['text1'].fillna('.').values
    train_label = train_data['label'].astype(int).values

    all_dataset = AugmentPairSamples(train_text_0, train_label, train_text_1)
    
    # Calculate Train/Validation ratio
    ratio = args.train_val_ratio
#     train_sample_ratio = math.ceil((len(all_dataset)*ratio))
#     val_sample_ratio = math.ceil(len(all_dataset)*(1-ratio))
    
#     print('train_sample_ratio', train_sample_ratio)
#     print('val_sample_ratio', val_sample_ratio)
    
#     train_dataset, val_dataset = random_split(all_dataset, [train_sample_ratio, val_sample_ratio],
#                                               generator=torch.Generator().manual_seed(42))

#     train_loader = util_data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
#     val_loader = util_data.DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4)

    
    if len(ratio) == 2:
        train_sample = math.ceil((len(all_dataset)*ratio[0]))
        val_sample = math.floor(len(all_dataset)*(ratio[1]))
#         val_sample2 = math.ceil(len(all_dataset)*(ratio[2]))
        
        print('train_sample', ratio[0], train_sample)
        print('val_sample', ratio[1], val_sample)
#         print('val_sample2', ratio[2], val_sample2)
        
#         if ratio[0] == 1.0:
        if ratio[0] != 1.0:
            
            train_dataset, val_dataset = random_split(all_dataset, [train_sample, val_sample],
                                                  generator=torch.Generator().manual_seed(42))
            
#             _, val_dataset = random_split(all_dataset, [len(all_dataset) - val_sample, val_sample],
#                                                   generator=torch.Generator().manual_seed(42))
            
#             _, val_dataset2 = random_split(all_dataset, [len(all_dataset) - val_sample2, val_sample2],
#                                                   generator=torch.Generator().manual_seed(42))
            
            train_loader = util_data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
            val_loader = util_data.DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=4)
#             val_loader2 = util_data.DataLoader(val_dataset2, batch_size=256, shuffle=True, num_workers=4)
         
        else:
            print('Train ratio should != 1.0')
            
    else:
        print('args.train_val_ratio should have 2 elements')
            
    
    return train_loader, val_loader


def train_unshuffle_loader(args):
    
##     W/o augmentation : read csv and set the name of headers
#     train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep='\t', names=["label", "text", "_"])
    
    # For Contextual
    train_data = pd.read_csv(os.path.join(args.data_path, args.dataname))
    
    # For Paraphraser
#     train_data = pd.read_csv(os.path.join(args.data_path, args.dataname), sep='\t')

    
##     W/o augmentation
#     train_data['text'] = train_data['text'].str.strip()
    
##     W/o augmentation : has only original text and label 
#     train_text = train_data['text'].fillna('.').values
#     train_label = train_data['label'].astype(int).values

##    W/ augmentation : original text and 1 augmented data
    train_text_0 = train_data['text0'].fillna('.').values
    train_label = train_data['label'].astype(int).values

##    W/o augmentation 
#     train_dataset = AugmentPairSamples(train_text, train_label)
    train_dataset = AugmentPairSamples(train_text_0, train_label)
    
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4) 
    return train_loader

def val_loader(args):
    train_data = pd.read_csv(os.path.join(args.data_path, args.dataname_val), sep='\t', names=["label", "text", "_"])
    train_data['text'] = train_data['text'].str.strip()
    train_text = train_data['text'].fillna('.').values
    train_label = train_data['label'].astype(int).values

    train_dataset = TextClustering(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)   
    return train_loader

