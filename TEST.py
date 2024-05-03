from openai import OpenAI
import os
import promptbench as pb
import csv
import tqdm
import anthropic

import pandas as pd
df_claude = pd.read_excel('database_claude.xlsx')
df_chatgpt = pd.read_excel('database.xlsx')

# Merge the two DataFrames based on the '1. prompt' and '3. text_sample' columns
merged_df = pd.merge(df_claude, df_chatgpt, on=['1. prompt', '2. type', '3. text_sample', '4. keywords'], suffixes=('_claude', '_chatgpt'))

# Iterate over each row in the merged DataFrame
total_response_len_chat = 0
total_response_len_claude = 0
for _, row in merged_df.iterrows():
    #add to responselen
    #print('claude response: ', row['6. output_claude'])
    total_response_len_claude += len(str(row['6. output_claude']))
    total_response_len_chat += len(str(row['6. output_chatgpt']))
    # Extract keywords from each professor's response

trlc = 0
trlg = 0
cc = 0
for cell in df_claude['6. output']:
    # Check if the cell contains a string
        trlc += len(cell)
        cc += 1

for cell in df_chatgpt['6. output']:
    # Check if the cell contains a string
        trlg += len(cell)
            


# Add the results as new columns to the merged DataFrame

# Find the top 5 most common keywords for each professor

# Calculate the average response length for each professor
avg_response_length_claude = merged_df['6. output_claude'].str.len().mean()
avg_response_length_chatgpt = merged_df['6. output_chatgpt'].str.len().mean()

print('avg response len, generated: claude: ', avg_response_length_claude, '; chatgpt: ', avg_response_length_chatgpt)
print("avg response len, calculated: claude: ", (total_response_len_claude / 211), "; chatgpt: ", (total_response_len_chat / 211))
print("avg response len, calculated, m2: claude: ", (trlc / cc), "; chatgpt: ", (trlg / cc))

"""
cross-reference -- look for trends in the responses from various models
"dont think its meaningful to look at the language model/who does the best", that'll change in the next month
worth looking at any trends across language models -- which do particularly well across models? are there any excerpts that none of the models do well on?

look for -- good prompts, bad exceprts

we want an excel or json file:
excerpt, prompt, language model, response, num keywords, final "does it pass"

TODO:
- add like 3 more inputs
- evaluate like 3 models - gpt-4, gemini(?), claude(?),???
- create graph showing binary efficacy of diff prompt types per model
- create graph showing keyword-per-response efficacy of diff prompt types per model
- create graph showing length-to-keyword-ratio efficacy of diff prompt types per model
-create graph of OVERALL (binary) EFFICACY OF MODELS



"what's the language model's favorite word?"""
