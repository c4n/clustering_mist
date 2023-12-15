"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from dataloader.dataloader import train_unshuffle_loader, val_loader
from sklearn import cluster
from sklearn.metrics import accuracy_score
import csv
from datetime import datetime


def prepare_task_input(model, batch, args, is_contrastive=False):
    if is_contrastive:
        # W/o Augmentation
#         text, class_label = batch['text'], batch['label'].cuda()
#         txts = [text]
     
       # W/o Augmentation
        text0, text1, class_label = batch['text0'], batch['text1'], batch['label'].cuda()
        txts = [text0, text1]
#         text0, text1, text2, class_label = batch['text0'], batch['text1'], batch['text2'], batch['label'].cuda()
#         txts = [text0, text1, text2]
        
        feat = []
        
        for text in txts:
            features = model.tokenizer.batch_encode_plus(text, max_length=args.max_length, return_tensors='pt', padding='longest', truncation=True)
            for k in features.keys():
                features[k] = features[k].cuda()
            feat.append(features)

        # W/o Augmentation
#        return feat[0], class_label.detach()
#         print(text0[0])
#         print(feat[0]['input_ids'][0])
        


        # W/ Augmentation
        return  feat, class_label.detach()
    
    else:
        text, class_label = batch['text0'], batch['label'].cuda()
        features = model.tokenizer.batch_encode_plus(text, max_length=args.max_length, return_tensors='pt', padding='longest', truncation=True)
        for k in features.keys():
            features[k] = features[k].cuda()
        return features, class_label.detach()

    return None

def evaluate_embedding(model, args, step, val_loader = None):
    confusion, confusion_model = Confusion(args.num_classes), Confusion(args.num_classes)
    model.eval()
    dataloader = train_unshuffle_loader(args)
    
    if(val_loader is not None):
        print('Validation from Val Split...')
        dataloader = val_loader
    
    print('---- {} evaluation batches ----'.format(len(dataloader)))

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            _, label = batch['text0'], batch['label'] 
            features, _ = prepare_task_input(model, batch, args, is_contrastive=False)
#             print('features.keys()', features.keys())
#             features.update({'args': args, 'batch': batch})
            
#             print('evaluate_embedding args:', features['args'])
            #embeddings = model.get_embeddings(features)
            args.is_augmented = False
            output_feature = model.forward(features, args = args, batch = batch)
            embeddings = output_feature['sentence_embedding']
            model_prob = model.get_cluster_prob(embeddings)

            if i == 0:
                all_labels = label
                all_embeddings = embeddings.detach()
                all_prob = model_prob
            else:
                all_labels = torch.cat((all_labels, label), dim=0)
                all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                all_prob = torch.cat((all_prob, model_prob), dim=0)
    
    all_pred = all_prob.max(1)[1]
    all_labels -= 1
    all_pred = all_pred.cpu()
#     torch.save(all_pred, 'all_pred.pt')
#     torch.save(all_labels, 'all_labels.pt')
#     print('all_pred.shape', all_pred.shape)
#     print('all_labels.shape',all_labels.shape)
    #print('all_pred', np.unique(all_pred, return_counts=True))
    print('all_pred', len(np.unique(all_pred, return_counts=True)[0]))

    #print('all_labels', np.unique(all_labels, return_counts=True))
    confusion_model.add(all_pred, all_labels)
    confusion_model.optimal_assignment(args.num_classes)
    acc_model = confusion_model.acc()
    acc_model_sklearn = accuracy_score(all_pred, all_labels)

    kmeans = cluster.KMeans(n_clusters=args.num_classes, random_state=args.seed)
    embeddings = all_embeddings.cpu().numpy()
    kmeans.fit(embeddings)
    pred_labels = torch.tensor(kmeans.labels_.astype(np.int))
    # clustering accuracy 
    confusion.add(pred_labels, all_labels)
    confusion.optimal_assignment(args.num_classes)
    acc = confusion.acc()
    acc_sklearn = accuracy_score(pred_labels, all_labels)
    
    ressave = {"acc":acc, "acc_model":acc_model}
    for key, val in ressave.items():
        args.tensorboard.add_scalar('Test/{}'.format(key), val, step)
    
    print('[Representation] Clustering scores:',confusion.clusterscores()) 
    print('[Representation] ACC: {:.4f}'.format(acc)) 
    print('[Representation] ACC sklearn: {:.4f}'.format(acc_sklearn)) 
    print('[Model] Clustering scores:',confusion_model.clusterscores()) 
    print('[Model] ACC: {:.4f}'.format(acc_model))
    print('[Model] ACC sklearn: {:.4f}'.format(acc_model_sklearn))
    
    # JSD_mm_wo_ln2
    # _stackoverflow
    with open('log_JSD_mm_wo_ln2_trash.csv', "a") as f:
        writer = csv.writer(f)
        time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        contrastive_local_loss = args.loss['contrastive_local_loss'].item()
        contrastive_global_loss = args.loss['contrastive_global_loss'].item()
        cluster_loss = args.loss['clustering_loss'].item()
        
#         row = [time, step, contrastive_loss, cluster_loss, acc, acc_model, confusion_model.clusterscores()['NMI'], args.note]
        row = [time, step, '{:.5f}'.format(contrastive_local_loss), '{:.5f}'.format(contrastive_global_loss), '{:.5f}'.format(cluster_loss), '{:.5f}'.format(acc), '{:.5f}'.format(acc_model), '{:.5f}'.format(confusion_model.clusterscores()['NMI']), args.note+'_lr=xx/-/xx scale=xx/xx batch= 256_']
        
        if f.tell() == 0:
            header = ['Time', 'Step', 'Contrastive Local Loss', 'Contrastive Global Loss', 'Clustering Loss', '[Representation] ACC', '[Model] ACC', 'NMI Model', 'Note: ...']
            writer.writerow(header)
        writer.writerow(row)
    
    return dataloader, pred_labels, all_labels





