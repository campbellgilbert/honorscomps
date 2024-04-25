from tqdm import tqdm
import json
import sys

# Add the directory of promptbench to the Python path
sys.path.append('/honorscomps/promptbench')
import promptbench as pb

print('All supported datasets: ')
print(pb.SUPPORTED_DATASETS)

#load dataset
#FIXME -- collate into 1 file?
f = open('GRADE_inputs.json')
dataset = json.load(f)
q = open('GRADE_prompts.json')
prompts = json.load(q)

# print all supported models in promptbench
print('All supported models: ')
print(pb.SUPPORTED_MODELS)

# load a model, flan-t5-large, for instance.
model = pb.LLMModel(model='gpt-3.5-turbo', max_new_tokens=300, temperature=0.0001)
#model = pb.LLMModel(model='gpt-4', max_new_tokens=300, temperature=0.0001)
#model = pb.LLMModel(model='gemini-pro', max_new_tokens=300, temperature=0.0001)
#model = pb.LLMModel(model='llama2-70b', max_new_tokens=300, temperature=0.0001)
#unforch looks like pb doesnt support mistral, claude, or most recent llama

"""
['google/flan-t5-large', 'llama2-7b', 'llama2-7b-chat', 'llama2-13b', 'llama2-13b-chat', 'llama2-70b', 'llama2-70b-chat', 'phi-1.5', 'phi-2', 'palm', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-1106-preview', 'gpt-3.5-turbo-1106', 'vicuna-7b', 'vicuna-13b', 'vicuna-13b-v1.3', 'google/flan-ul2', 'gemini-pro']
"""
#model = pb.LLMModel(model='llama2-13b-chat', max_new_tokens=10, temperature=0.0001)

def has_any(reply, keywords):
    return any(substring in reply for substring in keywords)

#1 if response contains ideal keywords, 0 if doesn't contain keywords
#METRICS TO MEASURE --
"""
- efficacy of different prompt types - expert, chain-of-thought, no-yapping
- performances of different models based on how binary keyword matching
- obviously efficiencies of different models but we're going to load that into its own set manually and then make some

GRAPHS. 
FUCK YOU.
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
for prompt in prompts:
    outputs = []

    predicted_keywords = []
    actual_keywords = []
    for data in tqdm(dataset):
        

        # process input
        input_text = prompt["prompt"] + " " + data['text_sample'] 

        #get output from model
        output = model(input_text)
        outputs.append(output)

        #keyword eval
        if(has_any(output, data['keywords'])):
            #a keyword exists, record binary keyword prevalence for prompttype
            ptes_KEYWORD_BINARY[prompt['type']] += 1
            
            #number-keyword eval
            numKeywords = sum(1 for substring in data['keywords'] if substring in output.lower())

            predicted_keywords.append(len(data['keywords']))
            actual_keywords.append(numKeywords)

            #keyword prevalence per prompt type
            #keyword-to-text ratio per prompt type -- length of output divided by number of keywords
            ptes_KEYWORD_RATIO[prompt['type']] += (numKeywords / len(output))

        total += 1
        #print("num keywords found: ", numKeywords)
        
    # evaluate
    #score = pb.Eval.compute_cls_accuracy(preds, labels)
    #print(f"{score:.3f}, {prompt}")

#now we want to load this to a file...


print("keyword predictions vs keyword actuals: ", sum(a == b for a, b in zip(predicted_keywords, actual_keywords)) / len(predicted_keywords))
print()
print("TOTAL NUM WITH KEYWORDS: ", sum())
print("TOTAL NUM: ", total)
print()
print("PROMPT TYPE EFFICACIES, binary keyword prevalence: ", ptes_KEYWORD_BINARY)
print("PROMPT TYPE EFFICACIES, keyword-to-text ratio: ", ptes_KEYWORD_RATIO)

print("good job campbell im so proud of you")