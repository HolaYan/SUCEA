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
    parser.add_argument("gpu_id", type=str, help="gpu id")
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    model_id = args.model

    llm = LLM(model=model_id, dtype=torch.float16, tensor_parallel_size=2)
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9,max_tokens=9000)

    generated = []
    gt = []

    whole_query_generated = json.load(open(args.dataset, 'r'))


    retrieve_rate = 0
    count = 0

    prompts = []
    for i in tqdm(range(len(whole_query_generated))):
        prompt = llm_rewrite_prompt(whole_query_generated[i]['text'])
        prompts.append(prompt)
        whole_query_generated[i]['ori text'] = whole_query_generated[i]['text']


    outputs = llm.generate(prompts, sampling_params)

    for i in range(len(outputs)):
        rewrite_text = outputs[i].outputs[0].text.strip().split('\n\n')[0]
        whole_query_generated[i]['text'] = rewrite_text

    json.dump(whole_query_generated,open(f'./paraphrase_data/{model_id}.praphrased.wo_kn.ori_query.json','w'),indent = 4)

