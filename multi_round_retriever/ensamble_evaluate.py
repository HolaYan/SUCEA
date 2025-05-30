import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy
from openai import OpenAI
import os
from tqdm import tqdm
import argparse


def gpt_choose_kn(kn,query,top_k,model_name):
    instr = f"This is RankLLM, an intelligent assistant that can rank passages based on their relevancy to the query. \nThe following are {len(kn)} passages, each indicated by number identifier []. I can rank them based on their relevance to query: {query}\n "

    end_instr = f"The search query is: {query} \nI will rank the {top_k} passages above based on their relevance to the search query. The passages will be listed in descending order using identifiers, and the most relevant passages should be listed f irst, and the output format should be [] > [] > etc, e.g., [1] > [2] > etc.\n The ranking results of the {top_k} passages (only identifiers) is:"

    for i in range(len(kn)):
        text = kn[i]['text']
        passage = f"[{i}] {text}\n"
        if len(end_instr) + len(instr) + len(passage) < 4096:
            instr += passage
    
    instr += end_instr

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
            "role": "system",
            "content": instr
            },

        ],
        max_tokens=4024,
    )

    result = response.choices[0].message.content
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('top_k', type=int, default=None)
    parser.add_argument('model_file_name', type=str, default=None)
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('model', type=str, default=None)
    args = parser.parse_args()

    top_k = args.top_k

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))
    
    model_id = args.model

    model_file_name = args.model_file_name

    # load data
    section_generated = json.load(open(args.dataset, 'r'))
    whole_query_generated = json.load(open(args.dataset, 'r'))

    print("R@",top_k)
    section_generated = json.load(open(f'./2M_result/{model_file_name}/200.split_sen_dict.json', 'r'))

    whole_query_generated = json.load(open('./2M_result/gpt3.5.200.split_sen.json', 'r'))

    ##### merge sections' kn #####
    merged_dict_section_generated = {}
    for dicto in tqdm(section_generated):
        
        id = dicto['id'].split('_')[0]
        
        top_k_list = []
        if id not in merged_dict_section_generated:
            merged_dict_section_generated[id] = dicto
            chosen_list = []
            check_sen = []
            merged_dict_section_generated[id]['split sections kn'] = []
            for tmp in dicto['retrieved kn']:
                if tmp['text'] not in check_sen and len(check_sen) < top_k:
                    chosen_list.append(tmp)
                    check_sen.append(tmp['text'])
                    merged_dict_section_generated[id]['split sections kn'].append(tmp)
            dicto['retrieved kn'] = chosen_list

        else:
            id_list = [tmp['id'] for tmp in merged_dict_section_generated[id]['split sections kn']]
            chosen_list = []
            check_sen = [dicto['text'] for dicto in merged_dict_section_generated[id]['split sections kn']]
            length = len(check_sen)
            for tmp in dicto['retrieved kn']:
                if tmp['text'] not in check_sen and len(merged_dict_section_generated[id]['split sections kn']) - length < top_k:
                    merged_dict_section_generated[id]['split sections kn'].append(tmp)
                    chosen_list.append(tmp)
                    check_sen.append(tmp['text'])
            dicto['retrieved kn'] = chosen_list

    merged_list_section_generated = []
    for id in merged_dict_section_generated:
        del merged_dict_section_generated[id]['retrieved kn']
        merged_dict_section_generated[id]['id'] = id
        merged_list_section_generated.append(merged_dict_section_generated[id])

    ##### merge section & whole query's kn #####
    all_k_evi = 0
    correct_evi = 0
    num_of_gt_evi = 0
    acc = 0
    ori_retrieve_rate = 0
    for i in tqdm(range(len(merged_list_section_generated))):
        
        gt_text = [dicto['text'] for dicto in whole_query_generated[i]['gold_evidence']]
        
        kn_list = [dicto['text'] for dicto in whole_query_generated[i]['retrieved kn'][:top_k]]

        chosen_kn = ' '.join(kn_list).split(' ')

        intersections = []
        for item in gt_text:
            gt_evi = item.split(' ')

            set1 = set(gt_evi)
            set2 = set(chosen_kn)

            intersection = set1.intersection(set2)

            result = list(intersection)
            if len(result)/len(gt_evi) > 0.8:
                intersections.append(item)

        if len(intersections) > 0:
            whole_query_generated[i]['ori retrieve result'] = 1
        else:
            whole_query_generated[i]['ori retrieve result'] = 0

        ori_retrieve_rate += whole_query_generated[i]['ori retrieve result']

        if merged_list_section_generated[i]['id'] == str(whole_query_generated[i]['id']):
            merged_kn = merged_list_section_generated[i]['split sections kn']
            check_sen = [tmp['text'] for tmp in merged_kn]
            length = len(check_sen)
            for tmp in whole_query_generated[i]['retrieved kn']:
                if tmp['text'] not in check_sen and len(merged_kn) - length < top_k:
                    merged_kn.append(tmp)
                    check_sen.append(tmp['text'])
                
            try:
                chosen_id = gpt_choose_kn(merged_kn[:15],' '.join(whole_query_generated[i]['split sentences'][:2]),top_k,model_id)
                numbers = re.findall(r'\d+', chosen_id)
                chosen_list = [merged_kn[int(i)] for i in numbers]
            except:
                pass

            whole_query_generated[i]['each sec retrieve kn'] = chosen_list

            ###### chosen top k kn acc ######
            gt_text = [dicto['text'] for dicto in whole_query_generated[i]['gold_evidence']]
        
            list_generated_text = [dicto['text'] for dicto in whole_query_generated[i]['each sec retrieve kn']]

            chosen_kn = ' '.join(list_generated_text).split(' ')

            intersections = []
            for item in gt_text:
                gt_evi = item.split(' ')

                set1 = set(gt_evi)
                set2 = set(chosen_kn)

                intersection = set1.intersection(set2)

                result = list(intersection)
                if len(result)/len(gt_evi) > 0.8:
                    intersections.append(item)

            if len(intersections) > 0:
                acc +=1

            ## recall
            correct_evi += len(intersections)
            num_of_gt_evi += len(gt_text)
            
            
    print('ori acc:',ori_retrieve_rate/len(whole_query_generated))
    print('Acc: ',acc/len(whole_query_generated))
    print('Recall: ',correct_evi/num_of_gt_evi)
    json.dump(whole_query_generated,open(f'./2M_result/tfidf_merged/{model_file_name}/top{top_k}.merged_sec.json','w'),indent = 4)
