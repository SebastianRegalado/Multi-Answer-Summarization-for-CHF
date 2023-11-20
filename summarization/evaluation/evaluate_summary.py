import json
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import torch

import argparse
from ref_free_metrics.supert import Supert
from nltk.tokenize import sent_tokenize
from utils.evaluator import evaluate_summary_rouge, add_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='SUPERT',
        description='Evaluate summaries using SUPERT',
    )

    parser.add_argument('input_path', type=str, help='path to the input file')
    args = parser.parse_args()

    output_path = args.input_path.replace('.json', '_scores.json')
    
    # pseudo-ref strategy: 
    # * top15 means the first 15 sentences from each input doc will be used to build the pseudo reference summary
    pseudo_ref = 'top15' 

    # read source documents
    with open(args.input_path, 'r') as f:
        data = json.load(f)

    # get unsupervised metrics for the summaries
    for d in tqdm(data):
        d["summary_score"] = {}
        for label in d["summary"]:
            input = d["summary_input"][label]
            sentences = sent_tokenize(input)
            summaries = [d["summary"][label]]
            source_docs = [(d['uri'], sentences)]
            supert = Supert(source_docs, ref_metric=pseudo_ref) 
            scores = supert(summaries)
            d["summary_score"][label] = scores[0]

    # write the scores to the output file
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    with open(output_path, 'r') as f:
        data = json.load(f)

    for label in ["information", "suggestion", "cause", "treatment", "experience"]:
        all_scores = [d["summary_score"][label] for d in data if label in d["summary"]]
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores)
        print(f"Label: {label}")
        print(f"Mean score: {mean_score:.4f} +/- {std_score:.4f}")
        print()

    # (Optional) compare the summaries against golden refs using ROUGE
    # if os.path.isdir('./rouge/ROUGE-RELEASE-1.5.5') and args.rouge:
    #     refs = reader.readReferences() # make sure you have put the references in data/topic_1/references
    #     summ_rouge_scores = []
    #     for summ in summaries:
    #         rouge_scores = {}
    #         for ref in refs:
    #             rs = evaluate_summary_rouge(summ, ref)
    #             add_result(rouge_scores, rs)
    #         summ_rouge_scores.append(rouge_scores)

    #     for mm in ['ROUGE-1', 'ROUGE-2','ROUGE-L','ROUGE-SU4']:
    #         rouge_scores = []
    #         for rs in summ_rouge_scores:
    #             rouge_scores.append( np.mean(rs[mm]) )
    #         print('reference-based',mm,'\n',rouge_scores)

