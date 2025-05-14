import json
from tqdm import tqdm
import random
from utils import write_json


random.seed(8)




def create_dataset(type = 'NQ'):
    TOTAL_DATASETS = []

    if type == 'SQuAD':
        '''
        Dict: [version, data]
            Str: version
            List: data
                Dict: [title, paragraphs]
                    Str: title
                    List: paragraphs
                        Dict: [qas, context]
                            List: qas
                                Dict: [question, id, answers, is_impossible]
                                    Str: question
                                    Str: id
                                    List: answers
                                        Dict: [text, answer_start]
                                            Str: text
                                            Num: answer_start
                                    Bool: is_impossible
                            Str: context
        '''
        with open("/home/liuxb/datasets/SQuAD/train-v2.0.json") as file:
            datas = json.load(file)['data']
        for data in tqdm(datas):
            for paragraph in data['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    new_dataset = {}
                    question = qa['question']
                    new_dataset['question'] = question
                    answers = qa['answers']
                    if len(answers)==0:
                        break
                    new_dataset['text'] = "Generate a question with <answer> as \'{}\' from the <context>: {}".format(answers[0]['text'], context)
                    TOTAL_DATASETS.append(new_dataset)

    elif type == 'NQ' or type == 'TQA':
        with open('/data3/liuxb/datasets/{}/{}_train_results.json'.format(type, type)) as files:
            datas = json.load(files)
        print("The number of {} dataset questions is {}".format(type, len(datas)))
        for data in tqdm(datas):
            question = data['question'] +  '?' if data['question'][-1]!='?' else data['question']
            words = question.split()
            words[0] = words[0].capitalize()
            question = " ".join(words)
            answer = data['answers'][0]
            text = []
            for ctx in data['ctxs']:
                if ctx['has_answer']:
                    text.append("Generate a question with <answer> as '{}' from the <context>: {}".format(answer, ctx['text']))
                        # 'Generate a question: <answer> {} <context> {}'.format(answer, ctx['text'])
                    break

            if len(text) > 0:
                for t in text:
                    TOTAL_DATASETS.append({'question': question, 'text': t})

    elif type == 'hotpot':
        '''
        List: datas
            Dict: [supporting_facts, level, question, context, answer, _id, type]
                List: supporting_facts
                    List: (context_title, index)
                Str: level (easy, medium, hard)
                Str: question
                List: context
                    List: (context_title, List: context)
                Str: answer
                Str: _id
                Str: type
        '''
        with open('/data/liuxb/datasets/HotpotQA/hotpot_train_v1.1.json') as fp:
            datas = json.load(fp)

        for data in tqdm(datas):
            try:
                if data['level']=='hard':
                    question = data['question']
                    answer = data['answer']
                    context_dict = {}
                    for c in data['context']:
                        context_dict[c[0]] = c[1]
                    context = []
                    for s_f in data['supporting_facts']:
                        context.append(context_dict[s_f[0]][s_f[1]])
                    t = ' '.join(context)

                    TOTAL_DATASETS.append({'question': question, 'text': "Generate a question with <answer> as '{}' from the <context>: {}".format(answer, t)})
            except Exception as e:
                print(e)

    else:
        raise Exception("Invalid type")

    return TOTAL_DATASETS


if __name__ == '__main__':

    total_datasets = create_dataset(type = 'hotpot')
    random.shuffle(total_datasets)
    print(total_datasets[0])
    print("The number of question_generator dataset is ", len(total_datasets))
    write_json(total_datasets, path='./datasets/hotpot_dataset.json')


    # with open('./datasets/TQA_dataset.json') as f:
    #     datas = json.load(f)
    # print(datas[1])
    # print(len(datas))
