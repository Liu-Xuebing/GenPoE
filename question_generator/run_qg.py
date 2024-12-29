import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
from questiongenerator import QuestionGenerator
from questiongenerator import print_qa
import json
from tqdm import tqdm
import numpy as np
import hydra


def write_json(x, path):
    with open(path, "w") as f:
        f.write(json.dumps(x, indent=4))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument("--show_answers", dest="show_answers", action="store_true", default=True)
    parser.add_argument("--use_qa_eval", dest="use_qa_eval", default=False)
    return parser.parse_args()



@hydra.main(config_path="../config", config_name="qg_config")
def generate_syn_knowledge(config):
    args = parse_args()
    qg = QuestionGenerator(config.model_dir)
    with open(config.test_dataset_file) as tp:
        datas = json.load(tp)
    with open(config.dataset_file) as dp:
        wiki_dict = json.load(dp)

    for data in datas:
        title = data['ctxs'][0]['title']['paragraph']
        paragraph_indexes = wiki_dict[title]
        print(paragraph_indexes)

    for ix, data in enumerate(tqdm(datas)):

        ctxs = data['ctxs'][:8]
        for ic, ctx in enumerate(ctxs):
            text = ctx['text']
            qa_list = qg.generate(

                text,
                num_questions=int(args.num_questions),
                use_evaluator=args.use_qa_eval)
            write_json(qa_list, "/data3/liuxb/code/MMOE/question_generator/syn_QA/NQ/{}_{}.json".format(ix, ic))
    print_qa(qa_list, show_answers=args.show_answers)



if __name__ == "__main__":
    generate_syn_knowledge()

