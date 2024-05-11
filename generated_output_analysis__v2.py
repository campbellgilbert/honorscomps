import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from collections import Counter
from textstat import flesch_reading_ease
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein import distance

#this version (HOPEFULLY) works for N many models

#load data
filenames = ['database_claude.xlsx', 'database_gpt4.xlsx']
models = ['Claude 3 Opus', 'GPT-4']
dfs = [pd.read_excel(file) for file in filenames]

#merge
merged_df = dfs[0]
for i in range(1, len(dfs)):
    merged_df = pd.merge(merged_df, dfs[i], on=['1. prompt', '2. type', '3. text_sample', '4. keywords'], suffixes=(f'_{models[i-1]}', f'_{models[i]}'))

# Load the keyword types from the JSON file
with open('keyword_groups.json') as f:
    keyword_types = json.load(f)

#functions 
def extract_keywords(text):
    words = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word.isalpha() and word not in stop_words]

#find semantically similar keywords, or synonyms
def find_similar_keywords(keyword, expected_keywords):
    similar_keywords = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            if lemma.name() in expected_keywords:
                similar_keywords.add(lemma.name())
    return list(similar_keywords)

#based on the feedback type list compiled, figure out what type of feedback is being offered here
def find_feedback_type(keywords):
    feedback_types = []
    for feedback_type, type_keywords in keyword_types.items():
        if any(keyword in type_keywords for keyword in keywords):
            feedback_types.append(feedback_type)
    return ', '.join(feedback_types)

#find jaccard similarity between two sets
def jaccard_similarity(set1, set2):
    union = len(set1.union(set2))
    intersection = len(set1.intersection(set2))
    return intersection / union if union != 0 else 0

def normalized_similarity(values):
    max_value = max(values)
    min_value = min(values)
    return [1 - ((value - min_value) / (max_value - min_value)) if max_value != min_value else 1 for value in values]

def normalized_similarity_lst(lists):
    min_vals = [min(lst) for lst in lists]
    max_vals = [max(lst) for lst in lists]
    min_val = min(min_vals)
    max_val = max(max_vals)
    
    normalized_lists = []
    for lst in lists:
        normalized_list = [1 - ((val - min_val) / (max_val - min_val)) if max_val != min_val else 1 for val in lst]
        normalized_lists.append(normalized_list)
    
    return np.mean(normalized_lists, axis=0)

#init lists to store the results for each response
total_sm_keywords_list = [[] for _ in range(len(dfs))]
total_sm_keywords_list_fordata = [[] for _ in range(len(dfs))]

common_keywords_list = []
expected_keywords_list = [[] for _ in range(len(dfs))]
similar_keywords_list = [[] for _ in range(len(dfs))]
flesch_kincaid_re_scores_list = [[] for _ in range(len(dfs))]
keyword_group_list = [[] for _ in range(len(dfs))]
overall_keywords = []


# Iterate over each row in the merged DataFrame
for _, row in merged_df.iterrows():
    keywords_list = []
    for i in range(len(dfs)):
        output = row[f'6. output_{models[i]}']
        #extract keywords from each model's response
        keywords = extract_keywords(output)
        keywords_list.append(set(keywords))

        #get total s-m keywords for output purposes
        total_sm_keywords_list[i].append(set(keywords))
        total_sm_keywords_list_fordata[i].append(keywords)

        #get total s-m keywords for top 10 purposes
        overall_keywords.extend(keywords)

        #find expected keywords
        expected_keywords_list[i].append(set([keyword for keyword in keywords if keyword in row['4. keywords']]))

        #find simliar keywords
        similar_keywords = []
        for keyword in keywords:
            similar_keywords.extend(set(find_similar_keywords(keyword, row['4. keywords'])))
        similar_keywords_list[i].append(', '.join(similar_keywords))

        #find f-k scores
        flesch_kincaid_re_scores_list[i].append(flesch_reading_ease(output))

        #find keyword group
        keyword_group_list[i].append(find_feedback_type(keywords))

    #get common keywords between models
    common_keywords = set.intersection(*keywords_list)
    common_keywords_list.append(', '.join(common_keywords))

"""PER-OUTPUT RESULTS"""
#add results as new columns to the merged df
for i in range(len(dfs)):
    merged_df[f'expected keywords {models[i]}'] = expected_keywords_list[i]
    merged_df[f'similar keywords {models[i]}'] = similar_keywords_list[i]
    merged_df[f'total S-M keywords {models[i]}'] = total_sm_keywords_list[i]
    merged_df[f'response type {models[i]}'] = keyword_group_list[i]
    merged_df[f'flesch kincaid_re {models[i]}'] = flesch_kincaid_re_scores_list[i]

merged_df['common keywords'] = common_keywords_list    

"""OVERALL RESULTS"""
#top 10 keywords, overall
top_overall_keywords = Counter(overall_keywords).most_common(10)

#top 10 keywords, per model, UNIQUE
top_keywords_permodel_list = []
for i in range(len(dfs)):
    flat_model_keywords = [item for sublist in total_sm_keywords_list_fordata[i] for item in sublist]

    top_total_keywords = Counter(flat_model_keywords).most_common(30)
    thismodel_keywords = [keyword for keyword in top_total_keywords if keyword[0] not in [item[0] for item in top_overall_keywords]]
    top_thismodel_keywords = [item[0][0] for item in Counter(thismodel_keywords).most_common(10)]
    top_keywords_permodel_list.append(top_thismodel_keywords)

#average response length, per model
avg_response_len_list = []
#top response type per model
top_response_types = []

for i in range(len(dfs)):
    output_words = merged_df[f'6. output_{models[i]}'].str.split().map(len)
    avg_response_len_list.append(output_words.mean())

    splitups = [word.strip() for string in keyword_group_list[i] for word in string.split(',')]
    top_response_types.append([phrase for phrase, _ in Counter(splitups).most_common(3)])

"""DIVERGENCE & CONVERGENCE"""
#find the most and least similar prompt and input

#STEP 1 -- find the divergence for the metrics measured so far
totalkeywords_difference_scores = []
flesch_kincaid_re_difference_scores = []
output_length_difference_scores = []
#and the lstein scores :P
lstein_scores = []

for i in range(len(dfs)):
    for j in range(i + 1, len(dfs)):
        # Calculate keyword similarity using Jaccard similarity
        keyword_similarity = merged_df.apply(lambda x: jaccard_similarity(x[f'total S-M keywords {models[i]}'], x[f'expected keywords {models[j]}']), axis=1)
        totalkeywords_difference_scores.append(keyword_similarity)

        # Normalize the Flesch-Kincaid Reading Ease scores
        flesch_kincaid_re_diff = abs(merged_df[f'flesch kincaid_re {models[i]}'] - merged_df[f'flesch kincaid_re {models[j]}'])
        min_fk_diff = flesch_kincaid_re_diff.min()
        max_fk_diff = flesch_kincaid_re_diff.max()
        flesch_kincaid_re_diff_normalized = (flesch_kincaid_re_diff - min_fk_diff) / (max_fk_diff - min_fk_diff)

        flesch_kincaid_re_difference_scores.append(flesch_kincaid_re_diff_normalized)

        # Normalize the response length difference
        response_length_diff = abs(merged_df[f'6. output_{models[i]}'].str.len() - merged_df[f'6. output_{models[j]}'].str.len())
        min_length_diff = response_length_diff.min()
        max_length_diff = response_length_diff.max()
        response_length_diff_normalized = (response_length_diff - min_length_diff) / (max_length_diff - min_length_diff)
        output_length_difference_scores.append(response_length_diff_normalized)

        #calculate the Levenshtein distances -- the higher, the more different
        lstein_score = merged_df.apply(lambda x: distance(x[f'6. output_{models[i]}'], x[f'6. output_{models[j]}']), axis=1)
        lstein_scores.append(lstein_score)


# Calculate the AVERAGE similarity score for each metric across all pairs of models
merged_df['totalkeywords_difference'] = sum(totalkeywords_difference_scores) / len(totalkeywords_difference_scores)
merged_df['flesch_kincaid_re_difference'] = [sum(scores) / len(scores) for scores in zip(*flesch_kincaid_re_difference_scores)]
merged_df['output_length_difference'] = [sum(scores) / len(scores) for scores in zip(*output_length_difference_scores)]
merged_df['levenshtein_distances'] = [sum(scores) / len(scores) for scores in zip(*lstein_scores)]

#calc similarity scores for each prompt/input independently

#find average for each metric
prompt_totalkeywords_scores = merged_df.groupby('1. prompt')['totalkeywords_difference'].mean()
prompt_fk_scores = merged_df.groupby('1. prompt')['flesch_kincaid_re_difference'].mean()
prompt_length_scores = merged_df.groupby('1. prompt')['output_length_difference'].mean()
prompt_lstein_scores = merged_df.groupby('1. prompt')['levenshtein_distances'].mean()

input_totalkeywords_scores = merged_df.groupby('3. text_sample')['totalkeywords_difference'].mean()
input_fk_scores = merged_df.groupby('3. text_sample')['flesch_kincaid_re_difference'].mean()
input_length_scores = merged_df.groupby('3. text_sample')['output_length_difference'].mean()
input_lstein_scores = merged_df.groupby('3. text_sample')['levenshtein_distances'].mean()

#find the average scores overall
prompt_similarity_scores = (prompt_fk_scores + prompt_totalkeywords_scores + prompt_length_scores + prompt_lstein_scores) / 4
input_similarity_scores = (input_fk_scores + input_totalkeywords_scores + input_length_scores + input_lstein_scores) / 4

#find prompt/input that are most different from mean
most_convergent_prompt = prompt_similarity_scores.idxmin()
most_convergent_input = input_similarity_scores.idxmin()

most_divergent_prompt = prompt_similarity_scores.idxmax()
most_divergent_input = input_similarity_scores.idxmax()


# Create a new DataFrame for the summary results
summary_df = pd.DataFrame({
    'Metric': 
              ['Top 10 keywords, Overall'] + 
              [f'Top 10 keywords, {models[i]}' for i in range(len(dfs))] + 
              [f'Average response length, {models[i]}' for i in range(len(dfs))] +
              [f'Most common response types, {models[i]}' for i in range(len(dfs))] +
              ['Most divergent input', 'Most divergent prompt',
               'Most convergent input', 'Most convergent prompt'] +
              [f'Predicted Keywords Per Output, {models[i]}, Average' for i in range(len(dfs))] +
              [f'Similar-Predicted Keywords, {models[i]}, Average' for i in range(len(dfs))] +
              [f'Semantically-Meaningful Keywords, {models[i]}, Average' for i in range(len(dfs))] +
              [f'F-K Reading Score, {models[i]}, Average' for i in range(len(dfs))] +
              ['Average Levenshtein distance'],
            
    'Value': [', '.join([keyword for keyword, _ in top_overall_keywords])] +
             [', '.join([keyword for keyword in modellist]) for modellist in top_keywords_permodel_list] +
             avg_response_len_list + 
             [top_response_types[i] for i in range(len(dfs))] +
             [most_divergent_input, most_divergent_prompt,
              most_convergent_input, most_convergent_prompt] +
             [merged_df[f'expected keywords {models[i]}'].str.len().mean() for i in range(len(dfs))] +
             [merged_df[f'similar keywords {models[i]}'].str.len().mean() for i in range(len(dfs))] +
             [merged_df[f'total S-M keywords {models[i]}'] .str.len().mean() for i in range(len(dfs))] +
             [merged_df[f'flesch kincaid_re {models[i]}'].mean() for i in range(len(dfs))] +
             [merged_df['levenshtein_distances'].mean()]
})
# Save the results to an Excel file
with pd.ExcelWriter('feedback_analysis__v2_FINAL.xlsx') as writer:
    merged_df.to_excel(writer, sheet_name='Detailed Results', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)