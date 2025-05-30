import base64
import os
import json
import re
from tqdm import tqdm
import argparse

def more_yes_or_no(word_list):
    yes_count = word_list.count('yes')
    no_count = word_list.count('no')

    if yes_count > no_count:
        return 'yes'
    elif no_count > yes_count:
        return 'no'

def cot_fact_verification(evidence,claim,model_name):
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
            "role": "system",
            "content": f"You are a well-informed and expert fact-checker. Here are some example of how to act as a professional fact-checker:\
            Claim Example 1: The Cantos is a poem with most of it written over a 40 plus year time span. \
            Evidences for claim example 1: \
            <Most of it was written between 1915 and 1962, although much of the early work was abandoned and the early cantos, as finally published, date from 1922 onwards. \n The Cantos by Ezra Pound is a long, incomplete poem in 116 sections, each of which is a canto. \n This thread then runs through the appearance of Kuanon, the Buddhist goddess of mercy, the moon spirit from Hagaromo (a Noh play translated by Pound some 40 years earlier), Sigismondo's lover Ixotta (linked in the text with Aphrodite via a reference to the goddess' birthplace Cythera), a girl painted by Manet and finally Aphrodite herself, rising from the sea on her shell and rescuing Pound/Odysseus from his raft.> \
            Step 1: the evidence indicates that The Cantos is a poem with most of it written over a 40 plus year time span.\nStep 2: the evidence also mentions that The Cantos by Ezra Pound is a long, incomplete poem in 116 sections.\nStep 3: the other pieces of evidence provided This thread then runs through the appearance of Kuanon, the Buddhist goddess of mercy, the moon spirit from Hagaromo (a Noh play translated by Pound some 40 years earlier).\n\nbased on the evidence provided, the claim that The Cantos is a poem with most of it written over a 40 plus year time span is supported. \nfinal rating: supported \
            Claim Example 2: One of the themes of All the King's Men is that journalism elevates ordinary men into unbiased observers of their time. \
            Evidences for claim example 2: \
            <One central motif of the novel is that all actions have consequences, and that it is impossible for an individual to stand aloof and be a mere observer of life, as Jack tries to do (first as a graduate student doing historical research and later as a wisecracking newspaperman). \n It is in this sense that the characters are \"all the king's men\", a line taken from the poem Humpty Dumpty (Penn biographer Joseph Blotner also notes, \"Like Humpty Dumpty, each of the major characters has experienced a fall of some kind\"). \n One central motif of the novel is that all actions have consequences, and that it is impossible for an individual to stand aloof and be a mere observer of life, as Jack tries to do (first as a graduate student doing historical research and later as a wisecracking newspaperman).> \
            Step 1: let's break down the claim and the evidences provided:\n\nclaim: One of the themes of All the King's Men - is that journalism elevates ordinary men - into unbiased observers of their time.\n\nevidences:\nOne central motif of the novel is that all actions have consequences, and that it is impossible for an individual to stand aloof and be a mere observer of life, as Jack tries to do (first as a graduate student doing historical research and later as a wisecracking newspaperman). - It is in this sense that the characters are \"all the king's men\", a line taken from the poem Humpty Dumpty (Penn biographer Joseph Blotner also notes, \"Like Humpty Dumpty, each of the major characters has experienced a fall of some kind\"). - One central motif of the novel is that all actions have consequences, and that it is impossible for an individual to stand aloof and be a mere observer of life, as Jack tries to do (first as a graduate student doing historical research and later as a wisecracking newspaperman).\n\nStep 2: based on the evidences provided, we can see that One of the themes of All the King's Men is indeed that journalism elevates ordinary men into unbiased observers of their time. Step 3: this directly contradicts the claim that it is impossible for an individual to stand aloof and be a mere observer of life. therefore, the claim is refuted.\n\nrating: refuted\" \
            Now its' your turn, you are provided with evidences regarding the following claim: {claim} \
            Evidences: \
            <{evidence}> \
            Based strictly on the main claim, and the evidences provided, you will provide: \
            rating: The rating for claim should be one of \"supported\" if and only if the Evidences specifically support the claim, \"refuted\" if and only if the Evidences specifically refutes the claim or \"failed\": if there is not enough information to support or refute the claim appropriately. \
            Is the claim: {claim} \"supported\", \"refuted\" or \"failed\" according to the available questions and answers? \
            Lets think step by step."
            },

        ],
        max_tokens=2048,
    )

    result = response.choices[0].message.content

    return result

def fact_verification(evidence,claim,model_name):
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
            "role": "system",
            "content": f"Determine whether the provided claim is consistent with the corresponding document. Consistency in this context implies that all information presented in the claim is substantiated by the document. If not, it should be considered inconsistent. \
            Document: [{evidence}] \
            Claim: [{claim}] \
            Please assess the claim\’s consistency with the document by responding with either \"yes\" \"no\" or \"unknown\". \
            Answer:"
            },

        ],
        max_tokens=1024,
    )

    result = response.choices[0].message.content

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('top_k', type=int, default=None)
    parser.add_argument('dataset', type=str, default=None)
    parser.add_argument('model', type=str, default=None)
    parser.add_argument('folder_name', type=str, default=None)
    args = parser.parse_args()

    top_k = args.top_k

    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))

    folder_name = args.folder_name

    model = args.model
    
    print(model)
    
    # load data
    all_dict = json.load(open(args.dataset,'r'))
    
    fact_acc = 0
    un_count = 0
    
    print('R@',top_k)
    for dicti in tqdm(all_dict):
        evidence_list = [dicto['text'] for dicto in dicti['each sec retrieve kn']]
        evidence = '\n'.join(evidence_list)

        try:
            claim = dicti['ori text']
        except:
            claim = dicti['text']

        try:
            cot_result = cot_fact_verification(evidence,claim,model).lower()
        except:
            cot_result = 'unknown'
        
        
        if 'refute' in cot_result:
            dicti['fact check'] = 'refutes'
        elif 'support' in cot_result:
            dicti['fact check'] = 'supports'
        else:
            dicti['fact check'] = 'unknown'
        
        if dicti['fact check'] == dicti['label'].lower():
            fact_acc += 1
        else:
            fact_acc += 0
        
        if dicti['fact check'] == 'unknown':
            un_count += 1

    print('unknown portion',un_count/len(all_dict))
    print('acc: ',fact_acc/len(all_dict))


    