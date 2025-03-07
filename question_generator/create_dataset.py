import json
from tqdm import tqdm
import random
from utils import write_json
import spacy
nlp = spacy.load("en_core_web_sm")

random.seed(8)


def is_complete_sentence(sentence):
    """判断一个句子是否完整（主+谓结构）"""
    doc = nlp(sentence)
    has_subject = any(token.dep_ in ("nsubj", "nsubjpass", "expl") for token in doc)
    has_verb = any(token.pos_ in ("VERB", "AUX") for token in doc)
    return has_subject and has_verb


def clean_paragraph_spacy(paragraph):
    """分句并删除首尾不完整的句子"""
    doc = nlp(paragraph)
    sentences = [sent.text for sent in doc.sents]
    # print(sentences)
    if len(sentences) > 1:
        # 只删除不完整的首句和尾句
        if not is_complete_sentence(sentences[0]):
            sentences = sentences[1:]
        if not is_complete_sentence(sentences[-1]):
            sentences = sentences[:-1]
    return sentences


def create_dataset(type = 'NQ'):
    TOTAL_DATASETS = []

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

    return TOTAL_DATASETS


if __name__ == '__main__':

    total_datasets = create_dataset(type = 'TQA')
    random.shuffle(total_datasets)
    print(total_datasets[0])
    print("The number of question_generator dataset is ", len(total_datasets))
    write_json(total_datasets, path='./datasets/TQA_dataset.json')


    # with open('./datasets/TQA_dataset.json') as f:
    #     datas = json.load(f)
    # print(datas[1])
    # print(len(datas))
