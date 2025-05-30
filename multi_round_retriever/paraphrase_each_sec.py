import regex as re
import logging
import argparse
import json
import time
import os
import string
from tqdm import tqdm
import copy
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))

def openai_word_replace_section(kn,query,model_name):
    fill_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
            "role": "system",
            "content": f'You should complete the sentence <{query}> with missing name (author, writer, owner, builder, award-winning titles etc.), number (date, year, population, acreage, etc), location and so on, replace the original adversarail point of view with detailed data or evidence (for example, the biggest population -> population 3,854,000; the date is unknown -> possible dates ranging from 1598 to 1608 ) and correct the conterfactual mistakes in sentence, which is done in order to avoid adversarial cases. You should never add too much new additional information to the sentence. Here is an example:\
            Given sentence: <The album was released in 2018.>\
            Expected result: The "Blackpink in Your Area" compilation album was released in 2018.\
            To help you better complete the requiring task, we provide following knowledge as contexts: <{kn}> \
            Now do the requiring task on the sentence: <{query}> based on the given knowledge.'
            },
        ],
        max_tokens=1024,
    )

    rewrite_sen = fill_response.choices[0].message.content
    
    return rewrite_sen

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('model', type=str, default=None)
    parser.add_argument('folder_name', type=str, default=None)
    args = parser.parse_args()

    model = args.model
    folder_name = args.folder_name

    filename = os.path.basename(args.dataset)

    # read all the data and store it
    all_data = json.load(open(args.dataset,'r'))
    
    new_all_data = []
    
    for i in tqdm(range(len(all_data[:200]))):
        dicto = all_data[i]
        kn_list = [dicto['text'] for dicto in dicto['each sec retrieve kn']]
        
        kn = ' '.join(kn_list)

        whole_rewrite_sen = []
        sections = dicto['split sentences']
        sen = ''
        gen_sen = ''
        for i in range(len(dicto['split sentences'])):
            sen = sections[i]
            gen_sen = openai_word_replace_section(kn,sen,model)
            whole_rewrite_sen.append(gen_sen)

        whole_rewrite_sen.append(dicto['text'])
        dicto['ori text'] = dicto['text']

        for i in range(len(whole_rewrite_sen)):
            gen_sen = whole_rewrite_sen[i]
            tmp = copy.copy(dicto)
            tmp['id'] = tmp['id'] + "_" + str(i)
            tmp['text'] = gen_sen
            tmp['generate path'] = whole_rewrite_sen[:i+1]
            new_all_data.append(tmp)

    file_name = (args.dataset).split('/')[-1]
    path_name = f"./2M_result/5iterations/{folder_name}/paraphrased2.per_section."+file_name

    print('Saving data in ',path_name)

    json.dump(new_all_data,open(path_name,'w'),indent = 4) 