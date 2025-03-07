from torch.utils.data import Dataset, DataLoader
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
import json
import torch
from torch.nn.utils.rnn import pad_sequence
import os
from utils import first_word_cap

class NQ_TQA_Dataset(Dataset):
    def __init__(self, config, tok, status, th=None):
        self.config = config
        self.tok = tok
        self.status = status
        self.data = []
        if status=='Train':
            if isinstance(th, list):
                for t in th:
                    f = os.path.join(self.config.train_path, t)
                    with open(f) as train_data:
                        datas = json.load(train_data)

                        sentence = datas['passage']
                        QAs = datas['QAs']
                        for QA in QAs:
                            question = QA['question'] + '?' if QA['question'][-1] != '?' else QA['question']
                            answer = QA['answer']
                            self.data.append([sentence, question, answer])
            else:
                f = os.path.join(self.config.train_path, th)
                with open(f) as train_data:
                    datas = json.load(train_data)
                for data in datas:
                    sentence = data['passage']
                    QAs = data['QAs']
                    for QA in QAs:
                        question = QA['question'] + '?' if QA['question'][-1] != '?' else QA['question']
                        answer = QA['answer']
                        self.data.append([sentence, question, answer])

        elif status == "Test":
            self.data = [th]

        else:
            raise AssertionError("Error state")



    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        if self.status == 'Train':
            passage, question, answer = self.data[idx]
            input_w_passage = 'Passage: {}\nQuestion: {}\nAnswer:'.format(passage, question)
            input_wo_passage = 'Question: {}\nAnswer:'.format(question)
            answer = '{}\n'.format(answer)
            tok_tuples_w_passage = self.tok_tuples(input_w_passage, answer)
            tok_tuples_wo_passage = self.tok_tuples(input_wo_passage, answer)
            return tok_tuples_w_passage, tok_tuples_wo_passage
        else:
            row = self.data[idx]
            question = row["question"] + "?" if row["question"][-1] != "?" else row["question"]
            question = first_word_cap(question)
            answers = row['answers']
            passage = row["ctxs"][:1]
            knowledge = '. '.join([p['text'] for p in passage])
            instruction = 'Base above knowledge, answer the following question with a very short phrase, such as “1998”, “May 16th, 1931”, or “James Bond”, to meet the criteria of exact match datasets.'
            prompt_1 = 'Question: {}\nAnswer:'.format(question)
            prompt_2 = 'Knowledge:\n{}\nQuestion: {}\nAnswer:'.format(knowledge, question)
            prompt_3 = 'Knowledge:\n{}\n{}\nQuestion: {}\nAnswer:'.format(knowledge, instruction, question)

            return (self.tok.encode(prompt_1, return_tensors="pt").cuda(),
                    prompt_1,
                    self.tok.encode(prompt_2, return_tensors="pt").cuda(),
                    prompt_2,
                    self.tok.encode(prompt_3, return_tensors="pt").cuda(),
                    prompt_3,
                    [question],
                    answers)



    def tok_tuples(self, prompt, answer):
        if isinstance(self.tok, GPT2TokenizerFast):
            answer = " " + answer
        else:
            answer = answer
        tok_prompt = self.tok(prompt, return_tensors="pt")
        tok_answer = self.tok(answer, return_tensors="pt", add_special_tokens=False)
        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)
        return tok_tuples


    def collate_fn(self, tuples):
        tokens_w = [item[0] for item in tuples]  # get w/ tokens
        tokens_wo = [item[1] for item in tuples]  # get wo tokens

        padded_tokens_w = {k: pad_sequence([t[k].squeeze(0) for t in tokens_w],
                                         batch_first=True,
                                         padding_value=-100 if k == "labels" else 0).cuda()
                         for k in tokens_w[0].keys()}

        padded_tokens_wo = {k: pad_sequence([t[k].squeeze(0) for t in tokens_wo],
                                         batch_first=True,
                                         padding_value=-100 if k == "labels" else 0).cuda()
                         for k in tokens_wo[0].keys()}
        return  padded_tokens_w, padded_tokens_wo


    def val_collate_fn(self, tuples):
        return (tuples[0][0],
                tuples[0][1],
                tuples[0][2],
                tuples[0][3],
                tuples[0][4],
                tuples[0][5],
                [t for t in tuples[0][6]],
                [t for t in tuples[0][7]])


def make_Training_loader(config, tok, th):
    train_set = NQ_TQA_Dataset(config, tok, status='Train', th=th)
    train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True, collate_fn=train_set.collate_fn)
    return train_loader

def make_Validation_loader(config, tok, th):
    valid_set = NQ_TQA_Dataset(config, tok, status='Test', th=th)
    valid_loader = DataLoader(valid_set, batch_size=config.valid_batch_size, shuffle=False, collate_fn=valid_set.val_collate_fn)
    return valid_loader