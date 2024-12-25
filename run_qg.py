import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from questiongenerator import QuestionGenerator
from questiongenerator import print_qa
import json
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer



def write_json(x, path):
    with open(path, "w") as f:
        f.write(json.dumps(x, indent=4))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument("--show_answers", dest="show_answers", action="store_true", default=True)
    parser.add_argument("--use_qa_eval", dest="use_qa_eval", default=False)
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    qg = QuestionGenerator('./training/t5-xl-question-generator')

    # with open('/data3/liuxb/datasets/NQ/NQ_test_rerank_results.json') as file:
    #     datas = json.load(file)
    # for ix, data in enumerate(tqdm(datas)):
    #     question = data["question"] + '?' if data["question"][-1]!="?" else data["question"]
    #     answer = data["answers"]
    #     ctxs = data['ctxs'][:8]
    #     for ic, ctx in enumerate(ctxs):
    #         text = ctx['text']
    #         qa_list = qg.generate(
    #
    #             text,
    #             num_questions=int(args.num_questions),
    #             use_evaluator=args.use_qa_eval)
    #         write_json(qa_list, "/data3/liuxb/code/MMOE/question_generator/syn_QA/NQ/{}_{}.json".format(ix, ic))
    # print_qa(qa_list, show_answers=args.show_answers)




    # --------------------------------------
    # --------------------------------------
    # --------------------------------------
    # create classification MoE
    # with open('/data3/liuxb/code/MMOE/question_generator/syn_QA/NQ/general_MOE.json') as file:
    #     datas = json.load(file)
    #
    # all = []
    # useful_datas = datas[:5000]
    # for ix, data in enumerate(tqdm(useful_datas)):
    #     text = data['text']
    #     qa_list = qg.generate(
    #         text,
    #         num_questions=int(args.num_questions),
    #         use_evaluator=args.use_qa_eval)
    #     all.extend(qa_list)
    # write_json(all, "/data3/liuxb/code/MMOE/question_generator/classification.json")
    # print_qa(qa_list, show_answers=args.show_answers)
    #
    #
    # sentences = []
    # model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    # with open("/data3/liuxb/code/MMOE/question_generator/classification.json") as f:
    #     datas = json.load(f)
    #     for data in tqdm(datas):
    #         sentences.append(data['question'])
    #
    # embeddings = model.encode(sentences)
    # np.save('/data3/liuxb/code/MMOE/question_generator/classification', embeddings)