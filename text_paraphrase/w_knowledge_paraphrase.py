import regex as re
import argparse
import json
import time
import os
import string
from tqdm import tqdm



masked_section_instr = "If the sentence is not a complicated complex clause, then complete the following requirements step by step.\
Split the sentence into subject, linking verb (like is was are...), predicate, and mask the predicate part with the string \"*****\". \
Then only return the masked sentence. \
If the sentence is a complicated complex clause, then also complete the following requirements step by step.\
Chose predicate part in the subordinate clause part and mask the part with \"*****\".\
Then only return the masked whole sentence."

rewrite_instr = "Complete the given sentence with phrases based on given knowledge. The phrases should be detail description and correct knowledge that are specific to what the sentence is describing (for example, gross for actor, owner name for buildings, award name for person, story of why use the name, assistance person name for  the event, time for the published work and so on, population data for country, exact number for data) \
Also replace the error details in the sentence. Then return the new sentence without any \"*\" symbol."

rewrite_gpt_set_instr = "Complete the given sentence with phrases based on 2018 Dec wikipedia knowledge. The phrases should be detail description and correct knowledge that are specific to what the sentence is describing (for example, gross for actor, owner name for buildings, award name for person, story of why use the name, assistance person name for  the event, time for the published work and so on, population data for country, exact number for data) \
Also replace the error details in the sentence. Then return the new sentence without any \"*\" symbol."


def check_consecutive_token_similarity(sentence1, sentence2):
    tokens1 = sentence1.split()
    tokens2 = sentence2.split()

    for i in range(len(tokens1) - 4):
        if tokens1[i:i+5] == tokens2[i:i+5]:
            return 1

    return 0
    
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

def openai_rerank_section(original_rank,query,model = 'gpt-3.5-turbo'):
    rerank_messages = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "system",
            "content": 'Based on the similarity with given query, rerank the given original rank knowledge by the order of similarity between query and each knowledge. And only return the reranked knowledge with new index.'
            },
            {
            "role": "user",
            "content": f"Query: {query}\nOriginal rank: {original_rank}"
            },

        ],
        max_tokens=1024,
    )

    masked_sen = rerank_messages.choices[0].message.content

    return rerank_kn

def openai_rewrite_section(new_rank,query,model = 'gpt-3.5-turbo'):
    mask_response = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "system",
            "content": masked_section_instr
            },
            {
            "role": "user",
            "content": f"Query: {query}"
            },

        ],
        max_tokens=1024,
    )

    masked_sen = mask_response.choices[0].message.content

    fill_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": rewrite_instr
            },
            {
            "role": "user",
            "content": f"Query: {query}\nKnowledge: {new_rank}"
            },

        ],
        max_tokens=1024,
    )

    rewrite_sen = fill_response.choices[0].message.content
    
    return masked_sen,rewrite_sen

def openai_gpt_set_rewrite_section(query,model = 'gpt-3.5-turbo'):
    mask_messages = [
        {
            "role": "system",
            "content": masked_section_instr
        },
        {
            "role": "user",
            "content": f"Query: {query}"
        }
    ]
    masked_sen = openai_chat_completion(messages=mask_messages, max_tokens=1024, model=model, temperature=1)

    mask_messages = [
        {
            "role": "system",
            "content": rewrite_gpt_set_instr
        },
        {
            "role": "user",
            "content": f"Query: {query}"
        }
    ]
    rewrite_sen = openai_chat_completion(messages=mask_messages, max_tokens=1024, model=model, temperature=1)
    
    return rewrite_sen

def openai_summarize(kn,model = 'gpt-3.5-turbo'):
    sum_response = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "system",
            "content": 'Summarize following knowledge, you should not change the main idea and key words in the knowlegde'
            },
            {
            "role": "user",
            "content": f"Knowledge: {kn}"
            },

        ],
        max_tokens=2048,
    )

    Summarization = sum_response.choices[0].message.content

    return Summarization

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()

    model = args.model # set the model name
    openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
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
        dicto['masked text'], dicto['text'] = openai_rewrite_section(kn,dicto['text'])
        
        
    save_path = (args.dataset).split('.json')[0] + f'.rewrite.top{k}kn.json'

    print('Saving data in ',save_path)
    json.dump(all_data,open(save_path,'w'),indent = 4) 