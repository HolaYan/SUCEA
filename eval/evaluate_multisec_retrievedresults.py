import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
from openai import OpenAI
import os
from tqdm import tqdm
import argparse
import copy

# choose_kn_instr = "Focus on the given query, select the top 10 knowledge most relevant to the query content. Then return the 10 knowledge as a list seperated by #"

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7F]+', '', text)

def check_consecutive_token_similarity(sentence1, sentence2):
    tokens1 = sentence1.split()
    tokens2 = sentence2.split()

    for i in range(len(tokens1) - 3):
        if tokens1[i:i+4] == tokens2[i:i+4]:
            return True

    return False

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
                matrix[i][j] = 1 + min(matrix[i - 1][j],      # delete
                                      matrix[i][j - 1],      # insert
                                      matrix[i - 1][j - 1])  # replace

    similarity1 = 1 - matrix[len_str1][len_str2] / max(len_str1, len_str2)
    # print(str1)
    if similarity1 >= 0.6:
        similarity = similarity1
    else:
        similarity2 = check_consecutive_token_similarity(" ".join(str1), " ".join(str2))

        similarity = similarity2
    return round(similarity,1) >= 0.9

class SimpleTokenizer(object):
    ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
    NON_WS = r'[^\p{Z}\p{C}]'

    def __init__(self):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
        )

    def tokenize(self, text, uncased=False):
        matches = [m for m in self._regexp.finditer(text)]
        if uncased:
            tokens = [m.group().lower() for m in matches]
        else:
            tokens = [m.group() for m in matches]
        return tokens

def _normalize(text):
    return unicodedata.normalize('NFD', text)

def has_answer(answers, text, tokenizer) -> bool:
    """Check if a document contains an answer string."""
    text = _normalize(text)
    text = tokenizer.tokenize(text, uncased=True)

    flag = 0
    for answer in answers:
        answer = _normalize(answer)
        answer = tokenizer.tokenize(answer, uncased=True)
        answer_flag = 0
        for i in range(0, len(text) - len(answer) + 1):
            substring = text[i: i + len(answer)]

            if calculate_similarity(substring, answer):
                return True
        
    return False

def gpt_choose_kn(kn,query):
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
            "role": "system",
            "content": choose_kn_instr
            },
            {
            "role": "user",
            "content": f"Query: {query}\Knowledge:{kn}"
            },

        ],
        max_tokens=1024,
    )

    

    result = response.choices[0].message.content
    
    return result

if __name__ == '__main__':
    generated = []
    gt = []

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('top_k', type=int, default=None)
    args = parser.parse_args()


    top_k = args.top_k
    print("R@",top_k)

    try:
        whole_query_generated = json.load(open(args.dataset, 'r'))
    except:
        whole_query_generated = []
        with open(args.dataset, 'r') as file:
            for line in file:
                data = json.loads(line)
                whole_query_generated.append(data)

    ##### merge section & whole query's kn #####
    new_all_data = {}
    for i in range(len(whole_query_generated)):
        id = whole_query_generated[i]['id'].split('_')[0] 
        gt_text = [dicto['text'] for dicto in whole_query_generated[i]['gold_evidence']]
        whole_query_generated[i][f'Step 2 top k kn'] = []
        kn_list = []
        try:
            for tmp in whole_query_generated[i]['retrieved kn']:
                if tmp['text'] not in kn_list and len(kn_list) < top_k:
                    kn_list.append(tmp['text'])
                    whole_query_generated[i][f'Step 2 top k kn'].append(tmp)
            del whole_query_generated[i]['retrieved kn']
        except:
            for tmp in whole_query_generated[i]['ctxs']:
                if tmp['text'] not in kn_list and len(kn_list) < top_k:
                    kn_list.append(tmp['text'])
                    whole_query_generated[i][f'Step 2 top k kn'].append(tmp)
            del whole_query_generated[i]['ctxs']

        step1_kn_list = [dicto['text'] for dicto in whole_query_generated[i]['each sec retrieve kn']]
        whole_query_generated[i][f'Step 1 top k kn'] = whole_query_generated[i]['each sec retrieve kn']
        del whole_query_generated[i]['each sec retrieve kn']

        ### step 1
        intersections = []
        for item in gt_text:
            gt_evi = item.split(' ')

            step1_kn = ' '.join(step1_kn_list).split(' ')

            set1 = set(gt_evi)
            set2 = set(step1_kn)

            intersection = set1.intersection(set2)

            result = list(intersection)
            if len(result)/len(gt_evi) > 0.8:
                intersections.append(item)


        if len(intersections) > 0:
            whole_query_generated[i]['step1 retrieve result'] = 1 
        else:
            whole_query_generated[i]['step1 retrieve result'] = 0
        
        whole_query_generated[i]['step1 retrieve gt number'] = len(intersections)
        whole_query_generated[i]['gold evidence number'] = len(gt_text)
        
        ### step 2
        intersections_step2 = []
        for item in gt_text:
            gt_evi = item.split(' ')

            chosen_kn = ' '.join(kn_list).split(' ')

            set1 = set(gt_evi)
            set2 = set(chosen_kn)

            intersection = set1.intersection(set2)

            result = list(intersection)
            if len(result)/len(gt_evi) > 0.8:
                intersections_step2.append(item)


        if len(intersections_step2) > 0:
            whole_query_generated[i]['step2 retrieve result'] = 1
        else:
            whole_query_generated[i]['step2 retrieve result'] = 0
        
        whole_query_generated[i]['step2 retrieve gt number'] = len(intersections_step2)

        if id not in new_all_data:
            new_all_data[id] = copy.copy(whole_query_generated[i])
            new_all_data[id]['selected_i_th'] = int(whole_query_generated[i]['id'].split('_')[1]) 
        else:
            if new_all_data[id]['step2 retrieve result'] == 1:
                continue
            else:
                new_all_data[id] = copy.copy(whole_query_generated[i])
                new_all_data[id]['selected_i_th'] = int(whole_query_generated[i]['id'].split('_')[1]) 

    step2_acc = 0
    step1_acc = 0
    all_data = []
    step1_recall = 0
    step2_recall = 0
    all_gold = 0
    for key in new_all_data:
        all_data.append(new_all_data[key])
        step2_acc += new_all_data[key]['step2 retrieve result']
        step1_acc += new_all_data[key]['step1 retrieve result']
        step1_recall += new_all_data[key]['step1 retrieve gt number']
        step2_recall += new_all_data[key]['step2 retrieve gt number']
        all_gold += new_all_data[key]['gold evidence number']

    print('Step1 Acc: ',step1_acc/len(all_data))  
    print('Step1 Recall: ',step1_recall/all_gold)        
    print('Step2 Acc: ',step2_acc/len(all_data))
    print('Step2 Recall: ',step2_recall/all_gold)
    import pdb;pdb.set_trace()
    file_name = (args.dataset).split('/')[-1]
    path_name = "data/tfidf/step2_rewrite_per_sec/gemma2_9b/tfidf_result/checked/"+file_name
    json.dump(all_data,open(path_name,'w'),indent = 4)

