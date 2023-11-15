import json
import sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import os

import argparse
from ref_free_metrics.supert import Supert
from utils.data_reader import CorpusReader
from utils.evaluator import evaluate_summary_rouge, add_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='SUPERT',
        description='Evaluate summaries using SUPERT',
    )

    parser.add_argument('input_path', type=str, help='path to the input file')
    parser.add_argument('output_path', type=str, help='path to the output file')
    args = parser.parse_args()
    
    # pseudo-ref strategy: 
    # * top15 means the first 15 sentences from each input doc will be used to build the pseudo reference summary
    pseudo_ref = 'top15' 

    reader = CorpusReader("data/topic_3")
    source_docs = reader.readSimpleDocs()
    summaries = reader.readSummaries()

    # read source documents
    with open(args.input_path, 'r') as f:
        data = json.load(f)

    source_docs_list = []
    summaries_list = []
    for d in data:
        source_docs = [(str(i+1), source_doc) for i, source_doc in enumerate(d['source_docs'])]
        source_docs_list.append(source_docs)
        summaries_list.append([d['summary']])

    # get unsupervised metrics for the summaries
    pbar = tqdm(total=len(data))
    for d, source_docs, summaries in zip(data, source_docs_list, summaries_list):
        supert = Supert(source_docs, ref_metric=pseudo_ref) 
        scores = supert(summaries)
        d["score"] = scores[0]
        pbar.update(1)

    pbar.close()

    # write the scores to the output file
    with open(args.output_path, 'w') as f:
        json.dump(data, f, indent=2)

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

