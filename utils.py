import json
import os
import torch
from nltk import pos_tag
from nltk.tokenize import word_tokenize
import torch.nn.functional as F
from collections import Counter
import numpy as np
from responses import replace

WHITESPACE_AND_PUNCTUATION = {' ', '.', ',', ':', ';', '!', '?', '$', '%', '(', ')', '[', ']', '-', '`', '\'', '"'}
ARTICLES = {'the', 'a', 'an'}


# files = os.listdir('/data3/liuxb/code/MMOE/question_generator/syn_QA/NQ_embedding')
# expert_embedding = {}
# for file in files:
#     if int(file.split('.')[0].split('_')[1]) >= 6:
#         continue
#     file_dir = os.path.join('/data3/liuxb/code/MMOE/question_generator/syn_QA/NQ_embedding', file)
#     loaded_embeddings = np.load(file_dir)
#     expert_embedding[file.split('.')[0]] = loaded_embeddings
# print('The number of Experts is {}'.format(len(expert_embedding)))
sub_dataset_expert = []
with open('/data3/liuxb/datasets/NQ/NQ_test_rerank_results.json') as f:
    datas =  json.load(f)
for data in datas[:1000]:
    ctx = data['ctxs'][:8]
    for c in ctx:
        if os.path.exists(os.path.join('/data3/liuxb/code/MMOE/question_generator/syn_QA/NQ_MoE', '{}.pth'.format(c['id']))):
            sub_dataset_expert.append(c['id'])
print(len(sub_dataset_expert))
sub_dataset_expert = list(set(sub_dataset_expert))
print(len(sub_dataset_expert))

def find_subsequence_index(source, target):
    for i in range(len(source) - len(target) + 1):
        if torch.equal(source[i:i + len(target)], target):
            return i  # Return the starting index if found
    return -1  # Return -1 if the subsequence is not found


def cal_anchor_embedding(sentences, model, tokenizer):
    average_embedding = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged_tokens = pos_tag(tokens)
        words = [word for word, pos in tagged_tokens if pos in ['NN', 'NNS', 'NNP', 'NNPS']]
        if len(words) > 0:
            inputs = tokenizer(words, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False)
            with torch.no_grad():  # 不需要梯度计算
                embeddings = model.model.embed_tokens(inputs['input_ids'].cuda())  # get the embedding by embed layer
            embeddings = embeddings[:, 0, :]
            average_embedding.append(torch.mean(embeddings, dim=0, keepdim=True))
        else:
            average_embedding.append(torch.zeros(size=(1,4096)).cuda())
    return torch.cat(average_embedding, dim=0)


def cross_entropy(
    logits: torch.FloatTensor,
    labels: torch.LongTensor
):
    if len(logits.shape) == 2:
        return F.binary_cross_entropy_with_logits(logits, labels)

    if len(logits.shape) == 3:
        ans_indice = torch.where(labels != -100)
        logits = logits[ans_indice]
        labels = labels[ans_indice]
        return F.cross_entropy(logits, labels)



def cal_EM_F1(predict: torch.FloatTensor, answers: torch.LongTensor):
    f1 = F1Single(answers, predict)
    if ExactMatchSingle(answers, predict):
        return 1, f1
    else:
        return 0, f1



def ExactMatchSingle(answers, predicted_answer):
    for ans in answers:
        if CleanAnswer(ans) == CleanAnswer(predicted_answer):
            return True
    return False



def F1Single(label_answer, predicted_answer):
    def GetTokens(text):
        text = CleanAnswer(text)
        for delimeter in WHITESPACE_AND_PUNCTUATION:
            text = text.replace(delimeter, ' ')
        return text.split()
    f1 = 0
    predicted_answer_tokens = Counter(GetTokens(predicted_answer))
    num_predicted_answer_tokens = sum(predicted_answer_tokens.values())
    for answer in label_answer:
        answer_tokens = Counter(GetTokens(answer))
        num_answer_tokens = sum(answer_tokens.values())
        num_same = sum((predicted_answer_tokens & answer_tokens).values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / num_predicted_answer_tokens
        recall = 1.0 * num_same / num_answer_tokens
        f1 = max(2 * precision * recall / (precision + recall), f1)
    return f1


def CleanAnswer(answer):
    answer = answer.strip().lower()
    answer = answer.replace(' , ',', ')
    answer = answer.replace(' - ','-')
    if isinstance(answer, str):
        answer = answer.replace(u'\u00a0', ' ')
    else:
        answer = answer.replace('\xc2\xa0', ' ')
    while len(answer) > 1 and answer[0] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[1:]
    while len(answer) > 1 and answer[-1] in WHITESPACE_AND_PUNCTUATION:
        answer = answer[:-1]

    answer = answer.split()
    if len(answer) > 1 and answer[0] in ARTICLES:
        answer = answer[1:]
    answer = ' '.join(answer)
    return answer


# def search_index(query, model):
#     embeddings = model.encode([query])
#     flag = -1
#     index = None
#     for key, value in expert_embedding.items():
#         similarities = model.similarity(embeddings, value)
#         final_mean_sim = torch.max(similarities)
#         if final_mean_sim > flag:
#             flag = final_mean_sim
#             index = key
#     return index


import torch
import torch.nn.functional as F  # 用于计算 softmax


# def search_topK_with_softmax(query, model, K):
#     embeddings = model.encode([query])
#     scores = []
#
#     for key, value in expert_embedding.items():
#         similarities = model.similarity(embeddings, value)
#         final_mean_sim = torch.max(similarities)
#         scores.append((final_mean_sim, key))
#
#     scores = sorted(scores, key=lambda x: x[0], reverse=True)
#     topK_scores = scores[:K]
#
#     sims = torch.tensor([item[0] for item in topK_scores])  # 相似度值
#     keys = [item[1] for item in topK_scores]  # keys
#
#     softmax_probs = F.softmax(sims, dim=0)
#
#     return keys, softmax_probs


def first_word_cap(text):
    words = text.split()
    words[0] = words[0].capitalize()
    text = " ".join(words)
    return text