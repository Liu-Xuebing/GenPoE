import json
from tqdm import tqdm
import random
from utils import write_json

random.seed(8)
TOTAL_DATASETS = []


def create_dataset(type = 'NQ'):
    if type == 'SQuAD':
        with open("/data3/liuxb/datasets/SQuAD/train-v2.0.json") as file:
            datas = json.load(file)['data']
        for data in tqdm(datas):
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    new_dataset = {}
                    question = qa['question']
                    new_dataset['question'] = question
                    answers = qa['answers']
                    if len(answers) == 0:
                        new_dataset['text'] = "Generate an unanswerable question based on the <context>. Ensure the question is relevant but has no answer in the text: {}".format(context)
                    else:
                        new_dataset['text'] = "Generate a question with <answer> as \'{}\' from the <context>: {}".format(answers[0]['text'], context)
                    TOTAL_DATASETS.append(new_dataset)

    if type == 'NQ' or type == 'TQA':
        with open('/data3/liuxb/datasets/{}/{}_train_results.json'.format(type, type)) as files:
            datas = json.load(files)
        print("The number of {} dataset questions is {}".format(type, len(datas)))

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


if __name__ == '__main__':

    total_datasets = create_dataset(type = 'SQuAD')
    random.shuffle(total_datasets)
    print([total_datasets[0]])
    print("The number of question_generator dataset is ", len(total_datasets))

    write_json(total_datasets, path='../datasets/question_generator/SQuAD_dataset.json')