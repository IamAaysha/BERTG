# BERTG: Graph Augmented Transformer for Text Classification
We propose the BERTG model that augments transformer based models with Graph Neural Networks (GNN). By combining the local structural relationship between input documents learnt by the GNN, with the contextual embeddings learnt by transformer based models, the proposed BERTG model outperforms existing baselines(BERT, GCN) in terms of precision, recall, and F1 scores across multiple classes, and for varying training data sizes. We demonstrate the performance of the BERTG model on the Amazon Reviews Dataset classification task, by comparing the performance with two baseline models: BERT and GCN. Our proposed architecture achieves improved performance using lesser training data proving both advantageous and efficient.

## [Dataset](./Dataset/Software_5.json.gz)

We use the“Software" category of the [Amazon reviews dataset](https://nijianmo.github.io/amazon/index.html)which has 12803 datapoints. We ignore all the user and rating metadata, and consider only the user’s text input for this project. The average length of the review is 175.98 words, with a total of 101170 unique words in the dataset. There are five different rating classes (1-5) in the dataset, and the distribution is: {1: 1500, 2: 719, 3: 1598, 4:
3016, 5: 5971}.
### Samples:
![image](https://user-images.githubusercontent.com/93538009/235259275-0add0de9-538b-4726-8e65-5d4368bb1f04.png)

![image](https://user-images.githubusercontent.com/93538009/235261694-dce6b6ce-4334-4272-a2e2-ee68421730b6.png)

## [Implementation](./BERTG.ipynb)
The proposed BERTG model captures both the local structural information, and the contextual information by combining the BERT and the GCN model respectively. The document node embeddings learnt by the GCN model is concatenated with the embeddings of the ‘[CLS]’ token from the last hidden state of the BERT model, before being passed to an MLP for classification. We analyse two different models: (1) BERT - Pretrained + GCN with Pretrained node initialization, and (2)BERT - Finetuned + GCN with Finetuned weight initialization.
The code in the [code notebook](./BERTG.ipynb) is organised to first show baseline BERT model, then how it is finetuned, graph creation, Baseline GCN model based on created graph using pretrained BERT embeddings, GCN model based on finetuned BERT Embeddings and finally the BERTG architecture that has Graph augmented BERT embeddings passed to an MLP for text classification

## Empirical analysis of our approach
<img width="846" alt="image" src="https://user-images.githubusercontent.com/93538009/235262555-7f100015-2afd-4670-88eb-5aec8e5cc6d1.png">

- The graph augmented BERTG model outperforms the BERT models, which shows that graph embeddings carry meaningful information useful for the text        classification task.
- BERT models outperform the GCN models on both pretrained and finetuned approaches, and across both the 10% and 30% training sample sizes. Therefore, contextual embeddings captures by BERT is more important than the local structural information that GCN captures.
- The difference between the accuracy score of the BERT Finetuned and BERTG Finetuned models is greater when only 10% training sample size (-0.0072)
compared to the 30% training sample size (-0.0039). This shows that graph embeddings are particularly helpful when less labelled data is available. Similarly for the F1 score as well.
- Across all the models, finetuned approach achieves better performance that the pretrained approach. 
- Across all the models the accuracy score is higher than the f1 score due to the imbalance in the class distribution of the dataset. As more data points
belong to the same class (4 and 5), achieving higher accuracy is easier than achieving a higher f1 score.

<div><img width="400" alt="image" src="https://user-images.githubusercontent.com/93538009/235262657-62e6d590-280a-4852-98e5-d283ee3248e3.png"> <img width="400" alt="image" src="https://user-images.githubusercontent.com/93538009/235262612-35748119-2709-441c-9975-b05b6d05e723.png"></div>
Thus, proposed Finetuned BERTG model outperforms all other baselines.

## Conclusion
In this project, we implement the BERTG model which captures both the contextual embeddings using a BERT model, and the local structural and sequential information using GCN for the text classification task. We show that the proposed model outperforms existing baselines on the Amazon Reviews Software dataset.
### Potential future works: 
- In this project, we consider the document-word graph as a homogeneous graph without any node types and edge types. Analysing the performance of a heterogeneous document-word graph is a potential futurework.
- Some document-word edges and all word-word edges were dropped to reduce the complexity of the graph. Analysing the performance of the model on the whole graph without removing any edges is a potential future work.
- In this project, the BERT embeddings and graph embeddings are directly concatenated. Adding attention based weights to balance between the BERT and graph embeddings can help in improving the model’s performance

