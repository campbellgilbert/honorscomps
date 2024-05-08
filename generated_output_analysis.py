import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from collections import Counter
from textstat import flesch_reading_ease
import json
#credit to claude opus 3 for the bulk of this code, outstanding work
# Load the Excel files into pandas DataFrames
df_claude = pd.read_excel('database_claude.xlsx')
df_chatgpt = pd.read_excel('database.xlsx')

# Merge the two DataFrames based on the similar columns -- ideally would've loaded this all into 1 excel sheet or found a better way of analyzing this without any excel sheets but hey man i'm on a deadline here
merged_df = pd.merge(df_claude, df_chatgpt, on=['1. prompt', '2. type', '3. text_sample', '4. keywords'], suffixes=('_claude', '_chatgpt'))

# Load the keyword types from the JSON file
with open('keyword_groups.json') as f:
    keyword_types = json.load(f)


#functions 
def extract_keywords(text):
    words = word_tokenize(str(text).lower())
    stop_words = set(stopwords.words('english'))
    return [word for word in words if word.isalpha() and word not in stop_words]

def find_similar_keywords(keyword, expected_keywords):
    similar_keywords = set()
    for syn in wordnet.synsets(keyword):
        for lemma in syn.lemmas():
            if lemma.name() in expected_keywords:
                similar_keywords.add(lemma.name())
    return list(similar_keywords)

def find_feedback_type(keywords):
    feedback_types = []
    for feedback_type, type_keywords in keyword_types.items():
        if any(keyword in type_keywords for keyword in keywords):
            feedback_types.append(feedback_type)
    return ', '.join(feedback_types)

def jaccard_similarity(set1, set2):
    set1 = set(set1.split(', '))
    set2 = set(set2.split(', '))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Initialize lists to store the results for each response
totalkeywords_claude = []
totalkeywords_chatgpt = []
common_keywords_list = []
expected_keywords_list = []
similar_keywords_list = []
flesch_kincaid_re_scores_claude = []
flesch_kincaid_re_scores_chatgpt = []
keyword_group_claude = []
keyword_group_chatgpt = []

# Iterate over each row in the merged DataFrame
for _, row in merged_df.iterrows():
    # Extract keywords from each model's response
    
    keywords_claude = extract_keywords(row['6. output_claude'])
    keywords_chatgpt = extract_keywords(row['6. output_chatgpt'])

    totalkeywords_claude.append(set(keywords_claude))
    totalkeywords_chatgpt.append(set(keywords_chatgpt))

    # Find common keywords between the models
    common_keywords = list(set(keywords_claude) & set(keywords_chatgpt))
    common_keywords_list.append(', '.join(common_keywords))
    
    # Find expected keywords in each model's response
    expected_keywords_claude = set([keyword for keyword in keywords_claude if keyword in row['4. keywords']])
    expected_keywords_chatgpt = set([keyword for keyword in keywords_chatgpt if keyword in row['4. keywords']])
    expected_keywords_list.append((', '.join(expected_keywords_claude), ', '.join(expected_keywords_chatgpt)))
    
    # Find similar keywords to expected keywords in each model's response
    similar_keywords_claude = []
    similar_keywords_chatgpt = []
    for keyword in keywords_claude:
        similar_keywords_claude.extend(set(find_similar_keywords(keyword, row['4. keywords'])))
    for keyword in keywords_chatgpt:
        similar_keywords_chatgpt.extend(set(find_similar_keywords(keyword, row['4. keywords'])))
    similar_keywords_list.append((', '.join(similar_keywords_claude), ', '.join(similar_keywords_chatgpt)))
    
    # Calculate Flesch-Kincaid Reading Ease scores for each model's response
    flesch_kincaid_re_scores_claude.append(flesch_reading_ease(str(row['6. output_claude'])))
    flesch_kincaid_re_scores_chatgpt.append(flesch_reading_ease(str(row['6. output_chatgpt'])))

    #aaaaaand find the keyword-group the responses belong to
    keyword_group_claude.append(find_feedback_type(keywords_claude))
    keyword_group_chatgpt.append(find_feedback_type(keywords_chatgpt))


# Add the results as new columns to the merged DataFrame
merged_df['total S-M keywords_claude'] = totalkeywords_claude
merged_df['total S-M keywords_chatgpt'] = totalkeywords_chatgpt

merged_df['common keywords'] = common_keywords_list

merged_df['expected keywords_claude'] = [keywords[0] for keywords in expected_keywords_list]
merged_df['expected keywords_chatgpt'] = [keywords[1] for keywords in expected_keywords_list]

merged_df['similar keywords_claude'] = [keywords[0] for keywords in similar_keywords_list]
merged_df['similar keywords_chatgpt'] = [keywords[1] for keywords in similar_keywords_list]

merged_df['response type_claude'] = keyword_group_claude
merged_df['response type_chatgpt'] = keyword_group_chatgpt

merged_df['flesch kincaid_re_claude'] = flesch_kincaid_re_scores_claude
merged_df['flesch kincaid_re_chatgpt'] = flesch_kincaid_re_scores_chatgpt

# Find the top 10 most common keywords for each model
top_keywords_claude = Counter(keyword for response in merged_df['6. output_claude'] for keyword in extract_keywords(response)).most_common(10)
top_keywords_chatgpt = Counter(keyword for response in merged_df['6. output_chatgpt'] for keyword in extract_keywords(response)).most_common(10)

# Calculate the average response length for each model
avg_response_length_claude = merged_df['6. output_claude'].str.len().mean()
avg_response_length_chatgpt = merged_df['6. output_chatgpt'].str.len().mean()


"""FIXME: REWRITE THIS SO IT FINDS THE MOST DIFFERENT ANSWERS SEMANTICALLY, NOT IN TERMS OF RESPONSE LENGTH"""
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


# Create a new DataFrame for the summary results
summary_df = pd.DataFrame({
    'Metric': ['Top 10 keywords - Claude 3 Opus', 'Top 10 keywords - GPT-4',
               'Average response length - Claude 3 Opus', 'Average response length - GPT-4',
               'Text sample with biggest divergence', 'Prompt with biggest divergence',
               'Text sample with most similar responses', 'Prompt with most similar responses', 
               'Predicted Keywords Per Output, Claude 3 Opus, Average', 'Predicted Keywords Per Output, GPT-4, Average',
               'Similar-Predicted Keywords, Claude 3 Opus, Average', 'Similar-Predicted Keywords, GPT-4, Average',
               'Semantically-Meaningful Keywords, Claude 3 Opus, Average', 'Semantically-Meaningful Keywords, GPT-4, Average',
               'F-K Reading Score, Claude, Average', 'F-K Reading Score, GPT-4, Average' 
               ],

    'Value': [', '.join([keyword for keyword, _ in top_keywords_claude]),
              ', '.join([keyword for keyword, _ in top_keywords_chatgpt]),
              avg_response_length_claude, avg_response_length_chatgpt,
              max_divergence_text_sample, max_divergence_prompt,
              max_similarity_text_sample, max_similarity_prompt,
              merged_df['expected keywords_claude'].str.len().mean(), merged_df['expected keywords_chatgpt'].str.len().mean(),
              merged_df['similar keywords_claude'].str.len().mean(), merged_df['similar keywords_chatgpt'].str.len().mean(),
              merged_df['total S-M keywords_claude'].str.len().mean(), merged_df['total S-M keywords_chatgpt'].str.len().mean(),
              merged_df['flesch kincaid_re_claude'].mean(), merged_df['flesch kincaid_re_chatgpt'].mean()
              ]
})
# Save the results to an Excel file
with pd.ExcelWriter('feedback_analysis_UPDATED_2.xlsx') as writer:
    merged_df.to_excel(writer, sheet_name='Detailed Results', index=False)
    summary_df.to_excel(writer, sheet_name='Summary', index=False)