# ref: https://github.com/facebookresearch/DrQA/blob/main/drqa/retriever/tfidf_retrieve.py
# python tfidf_retrieve.py /scratch/hl5775/work/retriever_code/fool-me-twice/dataset/test.jsonl --n-docs 10 --model data/wikipedia/fm2_kilt_w100_title-tfidf-ngram\=2-hash\=16777216-tokenizer\=simple.npz --doc-db data/wikipedia/fm2_kilt_w100_title.db
import regex as re
import logging
import argparse
import json
import time
import os
import string
import spacy

import torch.multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from drqa import retriever, tokenizers
from drqa.retriever import utils

from tqdm import tqdm

import numpy as np

from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def check_consecutive_token_similarity(sentence1, sentence2):
    tokens1 = sentence1.split()
    tokens2 = sentence2.split()

    for i in range(len(tokens1) - 4):
        if tokens1[i:i+5] == tokens2[i:i+5]:
            return 1

    return 0

def calculate_similarity(str1, str2):
    len_str1 = len(str1)
    len_str2 = len(str2)

    matrix = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]

    for i in range(len_str1 + 1):
        for j in range(len_str2 + 1):
            if i == 0:
                matrix[i][j] = j
            elif j == 0:
                matrix[i][j] = i
            elif str1[i - 1] == str2[j - 1]:
                matrix[i][j] = matrix[i - 1][j - 1]
            else:
                matrix[i][j] = 1 + min(matrix[i - 1][j],      # 删除
                                      matrix[i][j - 1],      # 插入
                                      matrix[i - 1][j - 1])  # 替换

    similarity1 = 1 - matrix[len_str1][len_str2] / max(len_str1, len_str2)
    # print(str1)
    # similarity = similarity1
    if round(similarity1,1) >= 0.9:
        similarity = similarity1
    else:
        similarity2 = check_consecutive_token_similarity(" ".join(str1), " ".join(str2))
        similarity = similarity2

    return round(similarity,1) >= 0.9
    
def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def remove_punctuation(input_string):
    punctuation = string.punctuation + "$"
    translator = str.maketrans('', '', punctuation)
    result_string = input_string.translate(translator)
    
    return result_string

def remove_content_in_parentheses(input_string):
    result_string = re.sub(r'\([^)]*\)', '', input_string)
    return result_string

PROCESS_TOK = None
PROCESS_DB = None

def init(tokenizer_class, tokenizer_opts, db_class, db_opts):
    global PROCESS_TOK, PROCESS_DB
    PROCESS_TOK = tokenizer_class(**tokenizer_opts)
    Finalize(PROCESS_TOK, PROCESS_TOK.shutdown, exitpriority=100)
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(
            pattern,
            flags=re.IGNORECASE + re.UNICODE + re.MULTILINE,
        )
    except BaseException:
        return False
    return pattern.search(text) is not None


def has_answer(answer, doc_id, match):
    """Check if a document contains an answer string.

    If `match` is string, token matching is done between the text and answer.
    If `match` is regex, we search the whole text with the regex.
    """
    global PROCESS_DB, PROCESS_TOK
    text = PROCESS_DB.get_doc_text(doc_id)
    # print(text)
    text = utils.normalize(remove_punctuation(remove_non_ascii(text)))
    # import pdb;pdb.set_trace()
    if match == 'string':
        # Answer is a list of possible strings
        # print("retrieved:",text)
        or_text = text
        text = PROCESS_TOK.tokenize(text.lower()).words(uncased=True)
        # 
        for single_answer in answer:
            answer_text = single_answer
            single_answer = utils.normalize(remove_punctuation((remove_non_ascii(single_answer.lower()))))
            # print("gt:",answer_text)
            single_answer = PROCESS_TOK.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                substring = text[i: i + len(single_answer)]
                if calculate_similarity(substring, single_answer):
                    return True
    elif match == 'regex':
        # Answer is a regex
        single_answer = utils.normalize(answer[0])
        if regex_match(text, single_answer):
            return True
    return False


def get_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    query, (doc_ids, doc_scores) = answer_doc
    # print('answer:',answer)
    for doc_id in doc_ids:
        # print(doc_id)
        if has_answer(query, doc_id, match):
            return 1
    # print("failed")
    return 0

def return_gt_score(answer_doc, match):
    """Search through all the top docs to see if they have the answer."""
    query, (doc_ids, doc_scores) = answer_doc
    # print('answer:',answer)
    top10_scores = []
    global PROCESS_DB, PROCESS_TOK
    top10_text = ''
    for index,doc_id in enumerate(doc_ids[:5]):
        text = PROCESS_DB.get_doc_text(doc_id)
        top10_text += text
        # score = calculate_cos_similarity(query, doc_id)
        # top10_scores.append(score)

    query = utils.normalize(remove_punctuation((remove_non_ascii(query.lower()))))
    # print("gt:",answer_text)
    query = PROCESS_TOK.tokenize(query)
    query = query.words(uncased=True)
    
    print(top10_text)
    top10_text = utils.normalize(remove_punctuation((remove_non_ascii(top10_text.lower()))))
    # print("gt:",answer_text)
    top10_text = PROCESS_TOK.tokenize(top10_text)
    top10_text = top10_text.words(uncased=True)

    overlap_score = top10_overlap_score(top10_text,query)

    return overlap_score

def top10_overlap_score(str1,str2):
    common_words = set(str1).intersection(set(str2))
    similarity = len(common_words) / len(str2)
    # print("topk kn: "," ".join(str1))

    return similarity

def retrieve_sect(all_data):
    # logger.info('Reading data ...')
    questions = []
    answers = []
    for data in all_data:
        question = data['text']
        questions.append(question)

    ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

    all_closest_docs = ranker.batch_closest_docs(
        questions, k=100, num_workers=args.num_workers ## !!!!修改wiki base时也需要修改对应数据
    )
    gt_docs = zip(questions, all_closest_docs)

    return all_closest_docs


if __name__ == '__main__':
    mp.set_start_method('spawn')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default='/scratch/hl5775/work/retriever_code/DrQA/models/tfidf.npz')
    parser.add_argument('--doc-db', type=str, default=None,
                        help='Path to Document DB')
    parser.add_argument('--tokenizer', type=str, default='regexp')
    parser.add_argument('--n-docs', type=int, default=10)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--match', type=str, default='string',
                        choices=['regex', 'string'])
    parser.add_argument('outfolder', type=str, default=None)
    args = parser.parse_args()

    # model = 'gpt-4-turbo'
    # openai.api_key = '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'
    filename = os.path.basename(args.dataset)

    # start time
    start = time.time()

    # read all the data and store it
    logger.info('Reading data ...')
    all_data = []
    all_data = json.load(open(args.dataset,'r'))
    

    closest_docs = retrieve_sect(all_data)
    db_class = retriever.DocDB
    db_opts = {'db_path': args.doc_db}
    PROCESS_DB = db_class(**db_opts)
    

    acc = 0
    for i in tqdm(range(len(all_data))):
        pair = closest_docs[i]
        (doc_ids, doc_scores) = pair
        dicto = all_data[i]
        dicto['retrieved kn'] = []

        for i in range(len(doc_ids)):
            # global PROCESS_DB
            doc_id = doc_ids[i]
            score = doc_scores[i]
            if len(dicto['retrieved kn']) < 100:
                
                text = PROCESS_DB.get_doc_text(doc_id)
                tmp = {'id': doc_id, 'text': text, 'score': score}
                dicto['retrieved kn'].append(tmp)
                
    json.dump(all_data,open(args.outfolder+filename,'w'),indent=4)
    
    
    
    