"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import numpy as np
from utils.metric import Confusion
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

def get_kmeans_centers(embedder, train_loader, num_classes):
    for i, batch in enumerate(train_loader):

        text, label = batch['text0'], batch['label']
        corpus_embeddings = embedder.encode(text)
        # print("corpus_embeddings", type(corpus_embeddings), corpus_embeddings.shape)
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings), axis=0)

    # Perform kmean clustering
    confusion = Confusion(num_classes)
    clustering_model = KMeans(n_clusters=num_classes)
    clustering_model.fit(all_embeddings)
    cluster_assignment = clustering_model.labels_

    true_labels = all_labels
    pred_labels = torch.tensor(cluster_assignment)    
    print("all_embeddings:{}, true_labels:{}, pred_labels:{}".format(all_embeddings.shape, len(true_labels), len(pred_labels)))

#     pred_labels += 1
    true_labels -= 1
    print('true_labels',true_labels)
    print('pred_labels',pred_labels)
#     print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, accuracy_score(true_labels, pred_labels), clustering_model.cluster_centers_.shape))
    confusion.add(pred_labels, true_labels)
    confusion.optimal_assignment(num_classes)
    print("Iterations:{}, Clustering ACC:{:.3f}, centers:{}".format(clustering_model.n_iter_, confusion.acc(), clustering_model.cluster_centers_.shape))
    
    return clustering_model.cluster_centers_



