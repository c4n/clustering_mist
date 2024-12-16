# MIST: Mutual Information Maximization for Short Text Clustering

This project reproduces the results of the MIST clustering method on the SearchSnippets dataset.

## Introduction

This is a clone for reproducing the results of the [MIST: Mutual Information Maximization for Short Text Clustering](https://aclanthology.org/2024.acl-long.610/) paper. The original code can be found [here](https://github.com/c4n/clustering_mist/).
The method is described in the paper "MIST: Mutual Information Maximization for Short Text Clustering".

<details>
<summary>Abstract</summary>
Short text clustering poses substantial challenges due to the limited amount of information provided by each text sample. Previous efforts based on dense representations are still inadequate as texts are not sufficiently segregated in the embedding space before clustering. Even though the state-of-the-art method utilizes contrastive learning to boost performance, the process of summarizing all local tokens to form a sequence representation for the whole text includes noise that may obscure limited key information. We propose Mutual Information Maximization Framework for Short Text Clustering (MIST), which overcomes the information drown-out by including a mechanism to maximize the mutual information between representations on both sequence and token levels. Experimental results across eight standard short text datasets show that MIST outperforms the state-of-the-art method in terms of Accuracy or Normalized Mutual Information in most cases.
</details>

## Test Environment

The following environment was used to reproduce the results:

*   **Python Version:** 3.9.21
*   **Libraries:** The required libraries and their versions are listed in `requirements.txt`. To install them, run:
    ```bash
    pip install -r requirements.txt
    ```
    ```
    annoy==1.17.3
    elasticsearch==8.16.0
    faiss==1.5.3
    flair==0.14.0
    gensim==4.3.0
    hnswlib==0.8.0
    nltk==3.8.1
    numpy==2.2.0
    pandas==1.5.3
    recommonmark==0.7.1
    Requests==2.32.3
    rich==13.9.4
    scikit_learn==1.4.2
    scikit_learn==1.2.2
    scipy==1.11.4
    setuptools==68.2.2
    Sphinx==5.0.2
    tensorboardX==2.6.2.2
    torch==2.5.1
    tqdm==4.65.0
    transformers==4.47.0
    ```
*   **Hardware:** NVIDIA GeForce RTX 3090
*   **Operating System:** Windows 10
*   **CUDA Version:** 11.8

Please ensure your environment matches these specifications for optimal results.

## Reproducing Detailed

### SearchSnippets Dataset
- File: `Reproducing_MIST_SearchSnippets.ipynb`.
- GPU Memory: 14~15 GB.
- Runtime: 8314.27 seconds.

### Citation

```bibtex
@inproceedings{kamthawee-etal-2024-mist,
    title = "{MIST}: Mutual Information Maximization for Short Text Clustering",
    author = "Kamthawee, Krissanee  and
      Udomcharoenchaikit, Can  and
      Nutanong, Sarana",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.610",
    doi = "10.18653/v1/2024.acl-long.610",
    pages = "11309--11324",
    abstract = "Short text clustering poses substantial challenges due to the limited amount of information provided by each text sample. Previous efforts based on dense representations are still inadequate as texts are not sufficiently segregated in the embedding space before clustering. Even though the state-of-the-art method utilizes contrastive learning to boost performance, the process of summarizing all local tokens to form a sequence representation for the whole text includes noise that may obscure limited key information. We propose Mutual Information Maximization Framework for Short Text Clustering (MIST), which overcomes the information drown-out by including a mechanism to maximize the mutual information between representations on both sequence and token levels. Experimental results across eight standard short text datasets show that MIST outperforms the state-of-the-art method in terms of Accuracy or Normalized Mutual Information in most cases.",
}
```