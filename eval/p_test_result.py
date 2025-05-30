import numpy as np
import scipy.stats as stats
import pandas as pd

def calculate_significance(acc, base_acc, n, num_classes):
    """
    calculate the significance of the improvement of the accuracy
    """
    # convert to ratio (0-1)
    p1 = acc / 100
    p0 = base_acc / 100
    
    # calculate the number of correct samples
    count1 = round(p1 * n)
    count0 = round(p0 * n)
    
    # ratio test
    p_pooled = (count0 + count1) / (2 * n)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (2/n))
    z_stat = (p1 - p0) / se
    p_value = 1 - stats.norm.cdf(z_stat)  # one-tailed test, test p1 > p0
    
    return z_stat, p_value

# dataset samples number
fm_samples =   # FOOLMETWICE samples
wice_samples =   # WICE samples

# FM has 2 classes, WICE has 3 classes
fm_classes = 3
wice_classes = 3

# extract table data
# 1. GPT-4o-mini
gpt_base_fm_contriever = 67.5
gpt_base_fm_tfidf = 61.5
gpt_base_wice_contriever = 33.7
gpt_base_wice_tfidf = 27.3

# systems and baseline accuracy
systems = [
    "ClaimDecompose", "QABrief", "ProgramFC", "MiniCheck", "OURS"
]

# load accuracy data from csv
accuracy_data = pd.read_csv('accuracy_data.csv')

# create result DataFrame
results = []

# calculate the significance of GPT-4o-mini
for i, system in enumerate(systems):
    # FM Contriever
    z_stat, p_val = calculate_significance(gpt_fm_contriever[i], gpt_base_fm_contriever, fm_samples, fm_classes)
    results.append({
        'Model': 'GPT-4o-mini', 
        'System': system, 
        'Dataset': 'FM', 
        'Retriever': 'Contriever',
        'Base_Acc': gpt_base_fm_contriever,
        'System_Acc': gpt_fm_contriever[i],
        'Improvement': gpt_fm_contriever[i] - gpt_base_fm_contriever,
        'Z_Stat': z_stat,
        'P_Value': p_val
    })
    
    # FM TFIDF
    z_stat, p_val = calculate_significance(gpt_fm_tfidf[i], gpt_base_fm_tfidf, fm_samples, fm_classes)
    results.append({
        'Model': 'GPT-4o-mini', 
        'System': system, 
        'Dataset': 'FM', 
        'Retriever': 'TFIDF',
        'Base_Acc': gpt_base_fm_tfidf,
        'System_Acc': gpt_fm_tfidf[i],
        'Improvement': gpt_fm_tfidf[i] - gpt_base_fm_tfidf,
        'Z_Stat': z_stat,
        'P_Value': p_val
    })
    
    # WICE Contriever
    z_stat, p_val = calculate_significance(gpt_wice_contriever[i], gpt_base_wice_contriever, wice_samples, wice_classes)
    results.append({
        'Model': 'GPT-4o-mini', 
        'System': system, 
        'Dataset': 'WICE', 
        'Retriever': 'Contriever',
        'Base_Acc': gpt_base_wice_contriever,
        'System_Acc': gpt_wice_contriever[i],
        'Improvement': gpt_wice_contriever[i] - gpt_base_wice_contriever,
        'Z_Stat': z_stat,
        'P_Value': p_val
    })
    
    # WICE TFIDF
    z_stat, p_val = calculate_significance(gpt_wice_tfidf[i], gpt_base_wice_tfidf, wice_samples, wice_classes)
    results.append({
        'Model': 'GPT-4o-mini', 
        'System': system, 
        'Dataset': 'WICE', 
        'Retriever': 'TFIDF',
        'Base_Acc': gpt_base_wice_tfidf,
        'System_Acc': gpt_wice_tfidf[i],
        'Improvement': gpt_wice_tfidf[i] - gpt_base_wice_tfidf,
        'Z_Stat': z_stat,
        'P_Value': p_val
    })

# calculate the significance of LLama-3.1-70B
for i, system in enumerate(systems):
    # FM Contriever
    z_stat, p_val = calculate_significance(llama_fm_contriever[i], llama_base_fm_contriever, fm_samples, fm_classes)
    results.append({
        'Model': 'LLama-3.1-70B', 
        'System': system, 
        'Dataset': 'FM', 
        'Retriever': 'Contriever',
        'Base_Acc': llama_base_fm_contriever,
        'System_Acc': llama_fm_contriever[i],
        'Improvement': llama_fm_contriever[i] - llama_base_fm_contriever,
        'Z_Stat': z_stat,
        'P_Value': p_val
    })
    
    # FM TFIDF
    z_stat, p_val = calculate_significance(llama_fm_tfidf[i], llama_base_fm_tfidf, fm_samples, fm_classes)
    results.append({
        'Model': 'LLama-3.1-70B', 
        'System': system, 
        'Dataset': 'FM', 
        'Retriever': 'TFIDF',
        'Base_Acc': llama_base_fm_tfidf,
        'System_Acc': llama_fm_tfidf[i],
        'Improvement': llama_fm_tfidf[i] - llama_base_fm_tfidf,
        'Z_Stat': z_stat,
        'P_Value': p_val
    })
    
    # WICE Contriever
    z_stat, p_val = calculate_significance(llama_wice_contriever[i], llama_base_wice_contriever, wice_samples, wice_classes)
    results.append({
        'Model': 'LLama-3.1-70B', 
        'System': system, 
        'Dataset': 'WICE', 
        'Retriever': 'Contriever',
        'Base_Acc': llama_base_wice_contriever,
        'System_Acc': llama_wice_contriever[i],
        'Improvement': llama_wice_contriever[i] - llama_base_wice_contriever,
        'Z_Stat': z_stat,
        'P_Value': p_val
    })
    
    # WICE TFIDF
    z_stat, p_val = calculate_significance(llama_wice_tfidf[i], llama_base_wice_tfidf, wice_samples, wice_classes)
    results.append({
        'Model': 'LLama-3.1-70B', 
        'System': system, 
        'Dataset': 'WICE', 
        'Retriever': 'TFIDF',
        'Base_Acc': llama_base_wice_tfidf,
        'System_Acc': llama_wice_tfidf[i],
        'Improvement': llama_wice_tfidf[i] - llama_base_wice_tfidf,
        'Z_Stat': z_stat,
        'P_Value': p_val
    })

# create result DataFrame
results_df = pd.DataFrame(results)

# add significance markers
def add_significance_markers(p_value):
    if p_value < 0.001:
        return "***"  # extremely
    elif p_value < 0.05:
        return "*"    # significant
    else:
        return ""     # not significant

results_df['Significance'] = results_df['P_Value'].apply(add_significance_markers)

# print results
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(results_df[['Model', 'System', 'Dataset', 'Retriever', 'Improvement', 'Z_Stat', 'P_Value', 'Significance']])

# save results to CSV
results_df.to_csv('significance_test_results.csv', index=False)

# print OURS system's results
ours_results = results_df[results_df['System'] == 'OURS']
print("Ours system's significance test results:")
print(ours_results[['Model', 'Dataset', 'Retriever', 'Improvement', 'Z_Stat', 'P_Value', 'Significance']])