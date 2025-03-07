import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import argparse
from questiongenerator import QuestionGenerator
from questiongenerator import print_qa
import json
from tqdm import tqdm
import hydra
import csv
from utils import read_json
from create_dataset import is_complete_sentence, clean_paragraph_spacy


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
    datas = read_json(config.test_dataset_file)

    random.seed(7)
    random.shuffle(datas)
    for data in tqdm(datas):
        ctxs = data['ctxs'][:4]
        for ctx in ctxs:
            if os.path.exists("/data3/liuxb/code/MMoE/syn_knowledge/TQA/{}.json".format(ctx['id'][len('wiki:'):])):
                continue
            qa_list = qg.generate(
                ctx['text'],
                num_questions=int(args.num_questions),
                use_evaluator=args.use_qa_eval)


            with open("/data3/liuxb/code/MMoE/syn_knowledge/TQA/{}.json".format(ctx['id'][len('wiki:'):]),
                      'w') as json_file:
                json.dump({'passage': ctx['text'], 'QAs': qa_list}, json_file, indent=4)  # indent用于格式化输出，使其更易读

        # print_qa(syn_knowledge, show_answers=args.show_answers)


if __name__ == "__main__":
    generate_syn_knowledge()
    # dir = '/data3/liuxb/code/MMoE/syn_knowledge/NQ'
    # files = os.listdir(dir)
    # print(len(files))
    # for f in files:
    #     with open(os.path.join(dir, f), 'r') as fp:
    #         data = json.load(fp)
    #     if len(data) == 0:
    #         print(f)
    # #         os.remove(os.path.join(dir, f))
