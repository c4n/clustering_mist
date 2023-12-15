"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
import torch
import numpy as np
# from utils.logger import statistics_log
from evaluation import prepare_task_input, evaluate_embedding
import time
import timeit
from datetime import datetime,timezone

def training(train_loader, learner, args, val_loader = None, val_loader2 = None, timer = timeit.default_timer()):
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
        if ( args.print_freq > 0 ) and (( i % args.print_freq == 0 ) or ( i == args.max_iter )):
#             statistics_log(args.tensorboard, losses=losses, global_step = i)
            
            if(args.train_val_ratio == -1): #NO Validation                
                evaluate_embedding(learner.model, args, i, val_loader)                
                
            elif(len(args.train_val_ratio) == 2):
                print('------------- Evaluate Training Set -------------')
#                 print('--------------------------------------------------------')
                evaluate_embedding(learner.model, args, i, train_loader)
                
                print('------------- Evaluate Validation Set -------------')
#                 print('--------------------------------------------------------')
                evaluate_embedding(learner.model, args, i, val_loader)
                
#                 print('2#------------- Evaluate Validation Set 2 -------------')
#                 print('--------------------------------------------------------')
#                 evaluate_embedding(learner.model, args, i, val_loader2)

            learner.model.train()
    
            end = timeit.default_timer()                
            now_utc = datetime.now(timezone.utc)
            print('Time UTC:', now_utc)
            print('Current running time', round(end - timer,2), 'seconds')
        
    return None   



             