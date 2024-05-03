from tqdm import tqdm
import json
import pandas as pd
#import promptbench as pb
import anthropic
import os

api_key = os.environ.get("ANTHROPIC_API_KEY")
#NO GIT PUSH UNTIL THIS API KEY IS REMOVED. I CANT GO BACK TO JAIL
client = anthropic.Client(api_key="")

#prep to add onto dataset

#load dataset
#FIXME -- collate into 1 file?
f = open('GRADE_inputs.json')
dataset = json.load(f)
q = open('GRADE_prompts.json')
prompts = json.load(q)

def has_any(reply, keywords):
    return any(substring in reply for substring in keywords)

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

results = []
#save what the LLMs give back -- do the LLMs actually give legit feedback?
#make sure you're not getting false positives/negatives

for prompt in prompts:
    outputs = []
    for data in tqdm(dataset):
        #godawful but promptbench doesn't have an anthropic slot so ah well.
        message = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=300,
            temperature=0.0001,
            system=prompt['prompt'],
            messages=[
                {"role": "user", "content": data['text_sample']}
            ]
        )
        output = " ".join(block.text for block in message.content)

        found_keywords = []
        num_keywords = 0
        # process input
        input_text = prompt["prompt"] + " " + data['text_sample'] 
        #get output from model
        outputs.append(output)

        #keyword eval
        if(has_any(output, data['keywords'])):
            #a keyword exists, record binary keyword prevalence for prompttype
            #FIXME: DEPRECATED on account of we know it doesn't have keywords if num_keywords is 0. duh.
            #ptes_KEYWORD_BINARY[prompt['type']] += 1
            
            found_keywords = [keyword for keyword in data['keywords'] if keyword in output.lower()]
            #number-keyword eval
            num_keywords = sum(1 for keyword in data['keywords'] if keyword in output.lower())

            #keyword prevalence per prompt type
            #keyword-to-text ratio per prompt type -- length of output divided by number of keywords
            ptes_KEYWORD_RATIO[prompt['type']] += (num_keywords / len(output))

        #print(prompt['prompt'], '\n', prompt['type'],'\n', data['text_sample'],'\n', data['keywords'],'\n', len(data['keywords']),'\n', output,'\n', found_keywords,'\n', num_keywords, '\n')

        results.append([
            prompt['prompt'], 
            prompt['type'],
            data['text_sample'],
            data['keywords'],
            len(data['keywords']),
            output,
            found_keywords,
            len(found_keywords)
        ])
        

#now we want to load this to a file...
df = pd.DataFrame(results, columns=['1. prompt','2. type','3. text_sample','4. keywords','5. num_keywords','6. output','7. found_keywords', '8. num_keywords'])

df.to_excel("database_claude.xlsx", index=False)
