import pandas as pd
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from collections import Counter
from textstat import flesch_reading_ease
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

#this version works for all 4 models not just the 2
#we're first going to test with just the data that we have

#load data
filenames = ['database_claude.xlsx', 'database_gpt4.xlsx']
models = ['Claude 3 Opus', 'GPT-4']
dfs = [pd.read_excel(file) for file in filenames]

#merge
merged_df = dfs[0]
for i in range(1, len(dfs)):
    merged_df = pd.merge(merged_df, dfs[i], on=['1. prompt', '2. type', '3. text_sample', '4. keywords'], suffixes=(f'_{i-1}', f'_{i}'))

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
    set1 = set(set1.split(', '))
    set2 = set(set2.split(', '))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def normalized_difference(values):
    max_value = max(values)
    min_value = min(values)
    return [(value - min_value) / (max_value - min_value) if max_value != min_value else 0 for value in values]

#init lists to store the results for each response
total_keywords_list = [[] for _ in range(len(dfs))]
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
        output = row[f'6. output_{i}']
        #extract keywords from each model's response
        keywords = extract_keywords(output)
        keywords_list.append(set(keywords))

        #get total keywords
        total_keywords_list[i].append(set(keywords))

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

        #for overall keywords
        overall_keywords.extend(keyword for response in merged_df[f'6. output_{i}'] for keyword in extract_keywords(response))

    #get common keywords between models
    common_keywords = set.intersection(*keywords_list)
    common_keywords_list.append(', '.join(common_keywords))

"""PER-OUTPUT RESULTS"""
#add results as new columns to the merged df
for i in range(len(dfs)):
    merged_df[f'expected keywords {models[i]}'] = expected_keywords_list[i]
    merged_df[f'similar keywords {models[i]}'] = similar_keywords_list[i]
    merged_df[f'total S-M keywords {models[i]}'] = total_keywords_list[i]
    merged_df[f'response type {models[i]}'] = keyword_group_list[i]
    merged_df[f'flesch kincaid_re {models[i]}'] = flesch_kincaid_re_scores_list[i]

merged_df['common keywords'] = common_keywords_list    


"""OVERALL RESULTS"""
#top 10 keywords, overall
top_overall_keywords = Counter(overall_keywords).most_common(10)

#top 10 keywords, per model, UNIQUE
top_keywords_permodel_list = []
for i in range(len(dfs)):

    top_keywords = Counter(word for word in overall_keywords if word not in top_overall_keywords).most_common(10)
    top_keywords_permodel_list.append(top_keywords)

#average response length, per model
avg_response_len_list = []
for i in range(len(dfs)):
    output_words = merged_df[f'6. output_{i}'].str.split().map(len)
    avg_response_len_list.append(output_words.mean())


"""DIVERGENCE & CONVERGENCE"""
vectorizer = TfidfVectorizer()
#find the most and least similar responses
#FIXME -- this array structure only works for 2 models. update to work for N many
cosine_scores = []
#this code (theoretically) works for N models as well
for r in range(len(merged_df)):
    #for each set of outputs for a question...
    vectors = vectorizer.fit_transform


    response_a = merged_df[f'6. output_{i}']
    a_vectors = vectorizer.fit_transform(response_a)
    for j in range(len(dfs)):
        if i != j:
            response_b = merged_df[f'6. output_{j}']
            b_vectors = vectorizer.fit_transform(response_b)

            cosine_similarity = 

            
    
    response = merged_df[f'6. output_{i}']

    tfidf_vectors = vectorizer.fit_transform(response)

"""
totalkeywords_dissimilarity = merged_df.apply(lambda x: 1 - jaccard_similarity(x[f'total S-M keywords_0'], x[f'total S-M keywords_1']), axis=1)

totalkeywords_similarity = merged_df.apply(lambda x: jaccard_similarity(x[f'total S-M keywords_0'], x[f'total S-M keywords_1']), axis=1)


response_type_dissimilarity = merged_df.apply(lambda x: int(x[f'response type_0'] != x[f'response type_1']), axis=1)

flesch_kincaid_re_dissimilarity = normalized_difference(merged_df[[f'flesch kincaid_re_{i}' for i in range(len(dfs))]].values.tolist())

output_length_dissimilarity = normalized_difference(merged_df[[f'6. output_{i}' for i in range(len(dfs))]].str.len().values.tolist())

# Calculate the overall dissimilarity score for each row
dissimilarity_score = totalkeywords_dissimilarity + response_type_dissimilarity + flesch_kincaid_re_dissimilarity + output_length_dissimilarity

# Find the input and prompt with the most different responses
most_different_input = merged_df.loc[merged_df['dissimilarity_score'].idxmax(), '3. text_sample']
most_different_prompt = merged_df.loc[merged_df['dissimilarity_score'].idxmax(), '1. prompt']


# Normalize the Flesch-Kincaid Reading Ease scores
merged_df['flesch_kincaid_re_diff'] = abs(merged_df['flesch kincaid_re_claude'] - merged_df['flesch kincaid_re_chatgpt'])
min_fk_diff = merged_df['flesch_kincaid_re_diff'].min()
max_fk_diff = merged_df['flesch_kincaid_re_diff'].max()
merged_df['flesch_kincaid_re_diff_normalized'] = (merged_df['flesch_kincaid_re_diff'] - min_fk_diff) / (max_fk_diff - min_fk_diff)

# Calculate keyword similarity using Jaccard similarity
merged_df['keyword_similarity'] = merged_df.apply(lambda x: jaccard_similarity(x['expected keywords_claude'], x['expected keywords_chatgpt']), axis=1)

# Normalize the response length difference
merged_df['response_length_diff'] = abs(merged_df['6. output_claude'].str.len() - merged_df['6. output_chatgpt'].str.len())
min_length_diff = merged_df['response_length_diff'].min()
max_length_diff = merged_df['response_length_diff'].max()
merged_df['response_length_diff_normalized'] = (merged_df['response_length_diff'] - min_length_diff) / (max_length_diff - min_length_diff)

# Calculate the composite dissimilarity score
merged_df['dissimilarity_score'] = merged_df['flesch_kincaid_re_diff_normalized'] + (1 - merged_df['keyword_similarity']) + merged_df['response_length_diff_normalized']

# Find the text sample and prompt with the highest dissimilarity score
max_divergence_text_sample = merged_df.loc[merged_df['dissimilarity_score'].idxmax(), '3. text_sample']
max_divergence_prompt = merged_df.loc[merged_df['dissimilarity_score'].idxmax(), '1. prompt']

# Calculate the composite similarity score
merged_df['similarity_score'] = (1 - merged_df['flesch_kincaid_re_diff_normalized']) + merged_df['keyword_similarity'] + (1 - merged_df['response_length_diff_normalized'])

# Find the text sample and prompt with the highest similarity score
max_similarity_text_sample = merged_df.loc[merged_df['similarity_score'].idxmax(), '3. text_sample']
max_similarity_prompt = merged_df.loc[merged_df['similarity_score'].idxmax(), '1. prompt']

"""
# Create a new DataFrame for the summary results
summary_df = pd.DataFrame({
    'Metric': 
              ['Top 10 keywords, Overall'] + 
              [f'Top 10 keywords, {models[i]}' for i in range(len(dfs))] + 
              [f'Average response length, {models[i]}' for i in range(len(dfs))] +
              ['Most divergent input', 'Most divergent prompt',
               'Most convergent input', 'Most convergent prompt'] +
              [f'Predicted Keywords Per Output, {models[i]}, Average' for i in range(len(dfs))] +
              [f'Similar-Predicted Keywords, {models[i]}, Average' for i in range(len(dfs))] +
              [f'Semantically-Meaningful Keywords, {models[i]}, Average' for i in range(len(dfs))] +
              [f'F-K Reading Score, {models[i]}, Average' for i in range(len(dfs))],

    'Value': [', '.join([keyword for keyword, _ in top_overall_keywords])] +
             [', '.join([keyword for keyword, _ in top_keywords]) for top_keywords in top_keywords_permodel_list] +
             avg_response_len_list +
             ["hello", "world",
              "hello", "world"] +
             [merged_df[f'expected keywords_{i}'].str.len().mean() for i in range(len(dfs))] +
             [merged_df[f'similar keywords_{i}'].str.len().mean() for i in range(len(dfs))] +
             [merged_df[f'total S-M keywords_{i}'].str.len().mean() for i in range(len(dfs))] +
             [merged_df[f'flesch kincaid_re_{i}'].mean() for i in range(len(dfs))]
})
# Save the results to an Excel file
with pd.ExcelWriter('feedback_analysis_UPDATED.xlsx') as writer:
    merged_df.to_excel(writer, sheet_name='Detailed Results', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)