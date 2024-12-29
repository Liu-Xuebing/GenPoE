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
            if pas['has_answer']:
                pos.append('title: {} context: {}'.format(pas['title'], pas['text']))
            else:
                neg.append('title: {} context: {}'.format(pas['title'], pas['text']))


        if len(pos) != 0 and len(neg) != 0:
            data_ = {"query": question,
                     "pos": pos,
                     "neg": neg,
                     "prompt": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'."}
            datas.append(data_)

    random.shuffle(datas)
    print("The total number of data is", len(datas))
    with open(output_dir, 'w') as fp:
        for entry in datas:
            json.dump(entry, fp)
            fp.write('\n')


if __name__ == '__main__':
    # create_dataset(output_dir='./datasets/reranker/NQ/reranker.json')
    with open('/data3/liuxb/code/MMoE/datasets/reranker/NQ/reranker.json', 'r', encoding='utf-8') as fp:
        data_list = []
        for line in fp:
            data = json.loads(line.strip())  # 将每行加载为字典
            print(data)
            exit()
            data_list.append(data)
