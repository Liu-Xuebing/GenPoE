import json
from tqdm import tqdm
import random
from utils import first_word_cap


def read_json(file_name):
    with open(file_name, 'r') as f:
        datas = json.load(f)
    return datas


def create_dataset(output_dir):
    NQs = read_json('/data3/liuxb/datasets/NQ/NQ_train_results.json')
    datas = []
    for nq in tqdm(NQs):
        question = nq['question'] + '?' if nq['question'][-1] != '?' else nq['question']
        question = first_word_cap(question)
        passages = nq['ctxs']
        pos, neg = [], []
        for pas in passages:
            if pas['has_answer'] and pas['title'] not in pos:
                pos.append(pas['title'])
        for pas in passages:
            if not pas['has_answer'] and pas['title'] not in pos and pas['title'] not in neg:
                neg.append(pas['title'])

        if len(pos) != 0 and len(neg) != 0:
            data_ = {"query": question,
                     "pos": pos,
                     "neg": neg,
                     "prompt": "Given a query A and a title B, determine whether the title is relevant to the query by providing a prediction of either 'Yes' or 'No'."}
            datas.append(data_)

    random.shuffle(datas)
    print("The total number of data is", len(datas))
    with open(output_dir, 'w') as fp:
        for entry in datas:
            json.dump(entry, fp)
            fp.write('\n')


if __name__ == '__main__':
    create_dataset(output_dir='./datasets/reranker/NQ/reranker.json')