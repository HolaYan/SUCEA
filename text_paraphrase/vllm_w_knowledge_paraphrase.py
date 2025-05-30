import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
from openai import OpenAI
import os
from tqdm import tqdm
import torch
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
import argparse


os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


def mixtral_rewrite_prompt(query):
    rewrite_instr = f"<s> [INST] This is RewriteLLM, an intelligent assistant that can rewrite sentences into a better one which can be easier to retrieve relevant knowledge. I can do this by paraphrasing the sentence or adding additional descriptions. \n[/INST] User: Now, proceed with the sentence {query} The rewrite result of the sentence (only rewriter) is:"
    
    
    return rewrite_instr

def llm_rewrite_prompt(query):
    rewrite_instr = f"This is RewriteLLM, an intelligent assistant that can rewrite sentences into a better one which can be easier to retrieve relevant knowledge. I can do this by paraphrasing the sentence or adding additional descriptions. \n\nNow, proceed with the sentence {query} The rewrite result of the sentence (only rewriter) is:"
    
    
    return rewrite_instr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    parser.add_argument('gpu_id', type=str, default=None)
    args = parser.parse_args()

    model = args.model # set the model name
    filename = os.path.basename(args.dataset)

    # start time
    start = time.time()

    # read all the data and store it
    print('Reading data ...')
    all_data = []
    all_data = json.load(open(args.dataset,'r'))
    
    
    k = 10 # set the number of  knowledge to use
    
    for i in tqdm(range(len(all_data))):
        dicto = all_data[i]
        # get the knowledge
        try:
            kn_list = [kn['text'] for kn in dicto['each sec retrieve kn'][:k]] 
        except:
            kn_list = [dicto['text'] for dicto in dicto['retrieved kn'][:k]]
        kn = ' '.join(kn_list)

        if len(kn) < 12800:
            kn = kn
        else:
            kn = openai_summarize(kn)
            print(1)
        
        dicto['original text'] = dicto['text']
        dicto['masked text'], dicto['text'] = llm_rewrite_prompt(kn,dicto['text'])
        
        
    save_path = (args.dataset).split('.json')[0] + f'.rewrite.top{k}kn.json'

    print('Saving data in ',save_path)
    json.dump(all_data,open(save_path,'w'),indent = 4) 
