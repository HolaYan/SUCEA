import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
from openai import OpenAI
import os
from tqdm import tqdm
import argparse

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
                matrix[i][j] = 1 + min(matrix[i - 1][j],      # 删除
                                      matrix[i][j - 1],      # 插入
                                      matrix[i - 1][j - 1])  # 替换

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
            "content": f"Query: {query}\nKnowledge:{kn}"
            },

        ],
        max_tokens=1024,
    )

    result = response.choices[0].message.content
    
    return result

if __name__ == '__main__':

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
    correct_evi = 0
    num_of_gt_evi = 0
    acc = 0
    for i in range(len(whole_query_generated[:200])):
        gt_text = [dicto['text'] for dicto in whole_query_generated[i]['gold_evidence']]
        try:
            step1_kn_list = [dicto['text'] for dicto in whole_query_generated[i]['retrieved kn'][:top_k]]
        except:
            step1_kn_list = [dicto['text'] for dicto in whole_query_generated[i]['ctxs'][:top_k]]

        intersections = []
        for item in gt_text:
            gt_evi = item.split(' ')
            chosen_kn = ' '.join(step1_kn_list).split(' ')

            set1 = set(gt_evi)
            set2 = set(chosen_kn)

            intersection = set1.intersection(set2)

            result = list(intersection)
            if len(result)/len(gt_evi) > 0.8:
                intersections.append(item)

        ## recall
        correct_evi += len(intersections)
        num_of_gt_evi += len(gt_text)
        
        if len(intersections) > 0:
            acc +=1
        else:
            import pdb;pdb.set_trace()
            
    print('Acc: ',acc/len(whole_query_generated[:200]))
    print('Recall: ',correct_evi/num_of_gt_evi)       
