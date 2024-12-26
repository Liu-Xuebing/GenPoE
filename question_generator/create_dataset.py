import json
from tqdm import tqdm
import random
from .utils import write_json

random.seed(8)
TOTAL_DATASETS = []


def create_dataset(type = ['NQ']):
    for t in type:
        with open('/data3/liuxb/datasets/{}/{}_train_results.json'.format(t, t)) as files:
            datas = json.load(files)
        print("The number of {} dataset questions is {}".format(t, len(datas)))


        for data in tqdm(datas):
            new_dataset = {}
            new_dataset['question'] = data['question'] +  '?' if data['question'][-1]!='?' else data['question']
            words = new_dataset['question'].split()
            words[0] = words[0].capitalize()
            new_dataset['question'] = " ".join(words)

            answer = data['answers'][0]
            for ctx in data['ctxs']:
                if ctx['has_answer']:
                    new_dataset['text'] = "Generate a question with <answer> as \'{}\' from the <context>: {}".format(answer, ctx['text'])
                        # 'Generate a question: <answer> {} <context> {}'.format(answer, ctx['text'])
                    break
            if 'text' in new_dataset:
                TOTAL_DATASETS.append(new_dataset)

    return TOTAL_DATASETS



total_datasets = create_dataset()
random.shuffle(total_datasets)
print([total_datasets[0]])
print("The number of question_generator dataset is ", len(total_datasets))

write_json(total_datasets, path='../datasets/NQ_TQA_dataset.json')