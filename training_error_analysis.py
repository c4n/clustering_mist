"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch
import numpy as np
from utils.logger import statistics_log
from evaluation_error_analysis import prepare_task_input, evaluate_embedding
import time
import pickle
import pandas as pd
from utils.metric import Confusion

def training(train_loader, learner, args, val_loader = None):
    print('\n={}/{}=Iterations/Batches'.format(args.max_iter, len(train_loader)))
    t0 = time.time()
    learner.model.train()
    count = 0
    for i in np.arange(args.max_iter+1):
#         print('loop:', i, 'batch count:', count)
        try:
            batch = next(train_loader_iter)  
        except:
            train_loader_iter = iter(train_loader)
            batch = next(train_loader_iter)
            count = 0
        
        feats, _ = prepare_task_input(learner.model, batch, args, is_contrastive=True)

        losses = learner.forward(feats, args, batch, use_perturbation=args.use_perturbation)
#         print('batch data len:', batch['text0'][0])
#         print("Note : loss = ",losses)
#         losses = learner(feats, use_perturbation=args.use_perturbation)
        args.loss = losses
#         count += 1
        if i == 1200 :
            statistics_log(args.tensorboard, losses=losses, global_step = i)
            dataloader, pred_label, all_label = evaluate_embedding(learner.model, args, i, val_loader)
            learner.model.train()            
            
            for j, batch in enumerate(dataloader):
                batch_text = batch['text0']
                if j == 0:
                    text0 = batch_text
                else:
                    text0.extend(batch_text)
                    
            ea = {'text0': text0,
                  'pred': pred_label,
                  'label': all_label                 
                 }


            
            confusion = Confusion(args.num_classes)
            confusion.add(ea['pred'], ea['label'])
            confusion.optimal_assignment(args.num_classes)
            
            label, pred = confusion.conf2label()
            
            df = pd.DataFrame(ea)
            df.label = label
            df.pred = pred
            
            df.to_csv('log_error_analysis/' + args.dataname + '.csv')
            with open('log_error_analysis/' + args.dataname +'.pickle', 'wb') as handle:
                pickle.dump(ea, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('Iteration:', i, 'Log saved!')
            break
        
        
    return None   



             