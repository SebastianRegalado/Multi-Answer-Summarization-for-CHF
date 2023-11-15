import sys
sys.path.append('../')
import numpy as np
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

    parser.add_argument('datapath', type=str, default='data/topic_3', help='path to the data folder')
    parser.add_argument('-r', '--rouge', action='store_true', help='whether to use ROUGE')
    args = parser.parse_args()
    
    # pseudo-ref strategy: 
    # * top15 means the first 15 sentences from each input doc will be used to build the pseudo reference summary
    pseudo_ref = 'top15' 

    # read source documents
    reader = CorpusReader(args.datapath)
    source_docs = reader.readSimpleDocs()
    summaries = reader.readSummaries()

    # get unsupervised metrics for the summaries
    supert = Supert(source_docs, ref_metric=pseudo_ref) 
    scores = supert(summaries)
    print('unsupervised metrics\n', scores)

    # (Optional) compare the summaries against golden refs using ROUGE
    if os.path.isdir('./rouge/ROUGE-RELEASE-1.5.5') and args.rouge:
        refs = reader.readReferences() # make sure you have put the references in data/topic_1/references
        summ_rouge_scores = []
        for summ in summaries:
            rouge_scores = {}
            for ref in refs:
                rs = evaluate_summary_rouge(summ, ref)
                add_result(rouge_scores, rs)
            summ_rouge_scores.append(rouge_scores)

        for mm in ['ROUGE-1', 'ROUGE-2','ROUGE-L','ROUGE-SU4']:
            rouge_scores = []
            for rs in summ_rouge_scores:
                rouge_scores.append( np.mean(rs[mm]) )
            print('reference-based',mm,'\n',rouge_scores)

