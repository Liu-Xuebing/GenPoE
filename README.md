### Question generator
The question generation part refers to the repository [question_generator](https://github.com/AMontgomerie/question_generator).

1. Generate the required training dataset using `./question_generator/create_dataset.py`.
2. Train the question generator. `./question_generator/qg_train.py`
3. Use the trained question generator. `./question_generator/run_qg.py`


### Re-ranker
We use the [**bge-reranker-v2-gemma**](https://huggingface.co/BAAI/bge-reranker-v2-gemma) as re-ranker base model in this work and extend our gratitude to its developers for their contributions to advancing re-ranking technologies.
1. Use the `./reranker/prepare_data.py` script to prepare the data for the re-ranker.
2. Training
3. After training, use the following code to merge model.
```bash
python -c "from FlagEmbedding.llm_reranker.merge import merge_llm; merge_llm('BAAI/bge-reranker-v2-gemma', model_dir, target_dir)"
```