import json
from openai import OpenAI
import os
from tqdm import tqdm
import time
import argparse

def split_section(sen,model_name):
    intr = f"This is SplitLLM, an intelligent assistant that can split sentences based on their semantic and syntactic structure within the sentence. The following are the sentences you need to split, indicated by number identifier []. I can split them based on their logical units, such as clauses, phrases, or specific grammatical boundaries. Please ensure that each unit retains its original meaning and context for better readability and understanding.\
    [0] {sen} \
    Provide the split sentences with each unit separated by a newline.\
    Example: \
    Original: \
    [0] The Natural is a book about Roy Hobbs a natural southpaw boxer who goes on to win the heavyweight title from Boom Boom Mancini.\
    Split: \
    [0] The Natural is a book | about Roy Hobbs | a natural southpaw boxer | who goes on to win the heavyweight | title from Boom Boom Mancini. \
    Now, proceed with the sentence: {sen} The split result of the sentence (only split) is: "

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": intr}
        ]
    )

    return response.choices[0].message.content

def section_to_sentence(sen,model_name):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": f'Based on the seperated sentence, if the section misses its subject, complete each split section with proper subject, then form a normal senrence containing enough details. If the the section is a complete sentence, remain the syntax. Here is an example for this split job:\
  <user> Given sentence: The film High Noon subverts gender norms of the time | by having the woman | rescue the man.\
  <response> The film High Noon subverts gender norms of the time. | High Noon unfolds by having the woman character. | The woman rescue the man in High Noon.'},
            {"role": "user", "content": f"Given Sentence: {sen}"}
        ]
    )

    return response.choices[0].message.content

# process the generated content into atomic sentences
def process_json(all_dict):
    section_dict = []
    split_sen_dict = []
    ids = []
    count = 0
    for dicti in all_dict:
        count = 0
        try:
            dicti['split section'] = dicti['split section'].split(']')[1].strip()
        except:
            pass
            
        for sec in dicti['split section'].split('|'):
            tmp = {}
            tmp['id'] = str(dicti['id']).split('_')[0] + '_' + str(count)
            tmp['text'] = sec.strip()
            tmp['whole sen'] = dicti['text']
            tmp['split section'] = dicti['split section']
            tmp['gold_evidence'] = dicti['gold_evidence']
            tmp['page'] = dicti['wikipedia_page']
            section_dict.append(tmp)
            count += 1
        
        count = 0
        for sen in dicti['split sentences']:
            tmp = {}
            tmp['id'] = str(dicti['id']).split('_')[0] + '_' + str(count)
            tmp['text'] = sen
            tmp['whole sen'] = dicti['text']
            # tmp['whole sen dpr result'] = dicti['dpr result']
            length = len(dicti['split section'].split('|'))
            tmp['split sentences'] = dicti['split sentences'][:length]
            tmp['gold_evidence'] = dicti['gold_evidence']
            tmp['page'] = dicti['wikipedia_page']
            # tmp['context'] = dicti['context']
            split_sen_dict.append(tmp)
            count += 1

    return section_dict, split_sen_dict

if __name__ == '__main__':
    # set argsparse to get the model name and other parameters if needed
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt-4o-mini-2024-07-18')
    parser.add_argument('--data_path', type=str, default='data/entailment_retrieval/claim/test.jsonl')
    args = parser.parse_args()
    

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "Your-OpenAI-API-Key"))
    
    model_id = args.model_name.split('-')[-1]
    model_name = args.model_name

    

    all_data = []
    for line in open(args.data_path):
        data = json.loads(line)
       
        all_data.append(data)
        
    for dicti in tqdm(all_data):
        
        for _ in range(5):
            try:
                dicti['split section'] = split_section(dicti['claim'],model_name)
                
                break
            except:
                time.sleep(5)
        
        for _ in range(5):
            try:
                sens = section_to_sentence(dicti['split section'],model_name)
                break
            except:
                time.sleep(5)

        dicti['split sentences'] = [sen.strip() for sen in sens.split('|')]
    
        json.dump(all_data,open(f'processed_data/{model_id}.split_sen.json','w'),indent=4) # all_data is the original data with split section and split sentences

        section_dict, split_sen_dict = process_json(all_data)

        json.dump(section_dict,open(f'processed_data/{model_id}.section_dict.json','w'),indent=4) # section_dict is the atomic section, each section may not be a complete sentence
        json.dump(split_sen_dict,open(f'processed_data/{model_id}.split_sen_dict.json','w'),indent=4) # split_sen_dict is the atomic sentence, each sentence is a complete sentence