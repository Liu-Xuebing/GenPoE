# GenPoE

This repository is based on the paper [GenPoE: Generalized Inner Product Operators for Parameter-Efficient Transfer Learning](https://openreview.net/forum?id=QRDLThn4jP).

### Retrieval & Data Source
The retrieval model uses the **DPR** model. The weights and NQ/TQA data can be obtained using the `./dpr/data/download.py` script. 

For details, please refer to the original [**DPR** repository](https://github.com/facebookresearch/DPR/tree/main).


### Training the Re-ranker

1. #### Prepare Training Data

Use the script `reranker/prepare_data` to prepare the training data.  
The data is constructed from the training sets of **NQ** and **TQA** into the following format:
```json
{
  "query": "question",
  "pos": "positive passage",  // Positive passages
  "neg": "negative passage",  // Negative passages
  "prompt": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."
 }
```

2. #### Train the Re-ranker
```python
python train_reranker.py
```

### Main part
![Main Figure](images/main.png)
