import json
from tqdm import tqdm
import random


def read_json(file_name):
    with open(file_name, 'r') as f:
        datas = json.load(f)
    return datas



def create_dataset(output_dir):
    NQs = read_json('NQ_train_results.json')
    datas = []
    for nq in tqdm(NQs):
        question = nq['question'] + '?' if nq['question'][-1] != '?' else nq['question']
        passages = nq['ctxs']
        pos, neg = [], []
        for pas in passages:
            if pas['has_answer']:
                pos.append(pas['text'])
            else:
                neg.append(pas['text'])

        if len(pos) == 0 or len(neg) == 0:
            continue

        data_ = {"query": question,
                 "pos": pos,
                 "neg": neg,
                 "prompt": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."}
        datas.append(data_)

    random.shuffle(datas)
    with open(output_dir, 'w') as fp:
        for entry in datas:
            json.dump(entry, fp)
            fp.write('\n')


if __name__ == '__main__':
    create_dataset(output_dir='')