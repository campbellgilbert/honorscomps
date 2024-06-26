from tqdm import tqdm
import json
import pandas as pd
import promptbench as pb

#prep to add onto dataset
"""try:
    existing_df = pd.read_excel("database.xlsx")
except FileNotFoundError:
    existing_df = pd.DataFrame()"""

#load dataset
#FIXME -- collate into 1 file?
f = open('GRADEE_inputs.json')
dataset = json.load(f)
q = open('GRADEE_prompts.json')
prompts = json.load(q)

# load a model
#model = pb.LLMModel(model='gpt-4', max_new_tokens=300, temperature=0.001)
model = pb.LLMModel(model='google/flan-ul2', max_new_tokens=300, temperature=0.0001)
#model = pb.LLMModel(model='llama2-70b', max_new_tokens=300, temperature=0.0001)
#unforch looks like pb doesnt support mistral, claude, or most recent llama
"""
google/flan-t5-large', 'llama2-7b', 'llama2-7b-chat', 'llama2-13b', 'llama2-13b-chat', 'llama2-70b', 'llama2-70b-chat', 'phi-1.5', 'palm', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'vicuna-7b', 'vicuna-13b', 'vicuna-13b-v1.3', 'google/flan-ul2
"""

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
        found_keywords = []
        num_keywords = 0
        # process input
        input_text = prompt["prompt"] + " " + data['text_sample'] 

        #get output from model
        output = model(input_text)
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

        #print("num keywords found: ", num_keywords)

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
        
    # evaluate
    #score = pb.Eval.compute_cls_accuracy(preds, labels)
    #print(f"{score:.3f}, {prompt}")

#now we want to load this to a file...

df = pd.DataFrame(results, columns=['1. prompt','2. type','3. text_sample','4. keywords','5. num_keywords','6. output','7. found_keywords', '8. num_keywords'])

df.to_excel("database_flanul2.xlsx", index=False)

"""
print("keyword predictions vs keyword actuals, avg: ", sum(a == b for a, b in zip(predicted_keywords, actual_keywords)) / len(predicted_keywords))
print()
print("TOTAL NUM WITH KEYWORDS: ", sum(1 for element in ptes_NUM_KEYWORDS if element != 0))
print("TOTAL NUM: ", total)
print()
print("PROMPT TYPE EFFICACIES, keyword prevalence: ", ptes_NUM_KEYWORDS)
print("PROMPT TYPE EFFICACIES, keyword-to-text ratio: ", ptes_KEYWORD_RATIO)

print("good job campbell im so proud of you")"""