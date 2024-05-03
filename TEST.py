from openai import OpenAI
import os
import promptbench as pb
import csv
import tqdm
import anthropic

import pandas as pd

#save what the LLMs give back -- do the LLMs actually give legit feedback?
#make sure you're not getting false positives/negatives
print("Supported models: ", pb.SUPPORTED_MODELS)
print("Supported datasets:")
try:
    existing_df = pd.read_excel("database.xlsx")
except FileNotFoundError:
    existing_df = pd.DataFrame()
# Corrected initial data structures
input_data = [{"text_sample": "peepeepoopoo", "keywords": "hello world"}]
prompting = [{"prompt": "hi hi hello world", "type": "yellow"}]

# Initialize an empty list to store the results
results = []

# Assuming output is intended to be used in the loop
output = "hello im mister frog this is my show i love you goodbye"

# Running the loop 3 times as per the given python script
for i in range(3):
    # Calculate the number of keywords in the output
    num_keywords = sum(1 for substring in input_data[0]['keywords'].split() if substring in output.lower())
    
    # Append the data to the results list in the required format
    results.append([
        prompting[0]['prompt'],
        prompting[0]['type'],
        input_data[0]['text_sample'],
        input_data[0]['keywords'],
        output,
        num_keywords
    ])

# Convert the results into a DataFrame
"""df = pd.DataFrame(results, columns=[
    'prompting[prompt]', 
    'prompting[type]', 
    'input[text_sample]', 
    'input[keywords]', 
    'output', 
    'num_keywords'])"""

# Convert the new results into a DataFrame
df = pd.DataFrame(results, columns=[
    'prompt', 
    'prompt type', 
    'input text sample', 
    'des. keywords', 
    'output', 
    'act. keywords'])

df.to_excel("database.xlsx", index=False)

# Combine existing data with new data
combined_df = pd.concat([existing_df, df], ignore_index=True)

# Output the combined DataFrame to an Excel sheet named "database.xlsx" in append mode
combined_df.to_excel("database.xlsx", index=False)

"""prompt = [{"text_sample": "peepeepoopoo", "keywords": "hello world"}]
input = [{"prompt": "hi hi hello world", "type": "yellow"}]

for i in range(3):
    num_keywords = sum(1 for substring in input['keywords'] if substring in output.lower())
    output = "hello i'm mister frog this is my show i love you goodbye"

#excel sheet tab, named for language file 
#prompt | prompt type || input_text | keywords || output | num kywrds in output


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

#testing -- putting outputs into an excel notebook
promptsy = [    
    {
        "prompt": "You are a high school English teacher giving feedback on a student's AP Language essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say 'Looks good!'. No yapping.: ",
        "type": "no-yapping"
    },
    {
        "prompt": "You are a high school English teacher giving feedback on a student's AP Language essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say 'Looks good!'. Get to the point.: ",
        "type": "no-yapping"
    }
] 

inputsy = [
    {
    "text_sample": "When I was younger, my parents unsuccessfully attempted to scare me into obeying them with threats of privileges being taken away or just being shut in my room. It didn't work. If anything, their trying to scare me into doing 'the right thing' only made me more determined to do the opposite. Any positive changes I could have made to be safer as a child were fought against solely because I didn't like being scared into doing things. I did indeed resent my parents for trying to control me through fear, and that kept any of the good they were attempting to enact from taking root. If reasoning and explanations were given when rules were told to me, perhaps I would have been more receptive. Instead, the fear instilled in me turned stronger and resentment towards the people who tried to change my mind, namely my parents.",
    "keywords": ["good"]
  },
  {
    "text_sample": "The way I see it, using fear as a tactic to get what you want can be useful and be resorted to if nothing else seems to work. Although it may be unenjoyable for some of the participating parties, it most likely will provide successful results.",
    "keywords": ["evidence", "reasoning", "nuance"]
  }
]

model = pb.LLMModel(model='gpt-3.5-turbo', max_new_tokens=300, temperature=0.000)
def has_any(reply, keywords):
    return any(substring in reply for substring in keywords)

#1 if response contains ideal keywords, 0 if doesn't contain keywords
#METRICS TO MEASURE --
"""
- efficacy of different prompt types - expert, chain-of-thought, no-yapping
- performances of different models based on how binary keyword matching
- obviously efficiencies of different models but we're going to load that into its own set manually and then make some
"""

#prompt type efficacies, binary keyword prevalence
ptes_KEYWORD_BINARY = {
    "chain-of-thought": 0,
    "expert": 0,
    "no-yapping": 0,
    "baseline": 0,
}

#prompt type efficacies, number of keywords
ptes_NUM_KEYWORDS = {
    "chain-of-thought": 0,
    "expert": 0,
    "no-yapping": 0,
    "baseline": 0,
}

#prompt type efficacies, keyword-to-text ratio
ptes_KEYWORD_RATIO = {
    "chain-of-thought": 0,
    "expert": 0,
    "no-yapping": 0,
    "baseline": 0
}

hasKeywords = 0
total = 0