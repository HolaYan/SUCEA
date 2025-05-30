import json
from openai import OpenAI
import os
from tqdm import tqdm

def prompt_rewrite(query,model_name):
    rewrite_instr = f"This is RewriteGPT, an intelligent assistant that can rewrite sentences into a better one which can be easier to retrieve relevant knowledge. I can do this by paraphrasing the sentence or adding additional descriptions. Now, proceed with the sentence {query} The rewrite result of the sentence (only rewriter) is:"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
            "role": "system",
            "content": rewrite_instr
            },
        ],
        max_tokens=254,
    )

    result = response.choices[0].message.content
    
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo')
    args = parser.parse_args()

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY"))

    whole_query_generated = json.load(open(args.dataset, 'r')) # load the data
    model_name = args.model # set the model name

    retrieve_rate = 0
    count = 0

    ##### GPT rewrite wo kn #####
    for i in tqdm(range(len(whole_query_generated))):
        for _ in range(5):
            try:
                rewrite_text = prompt_rewrite(whole_query_generated[i]['text'],model_name)
                
                break
            except:
                continue

        whole_query_generated[i]['ori text'] = whole_query_generated[i]['text']
        whole_query_generated[i]['text'] = rewrite_text

        json.dump(whole_query_generated,open(f'./paraphrase_data/{model_name}.praphrased.wo_kn.ori_query.json','w'),indent = 4)


