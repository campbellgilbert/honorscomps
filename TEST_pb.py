import promptbench as pb
from tqdm import tqdm

# print all supported datasets in promptbench
#print('All supported datasets: ')
#print(pb.SUPPORTED_DATASETS)

# load a dataset, sst2, for instance.
# if the dataset is not available locally, it will be downloaded automatically.
#dataset = pb.DatasetLoader.load_dataset("sst2")
#how does this work? what does it do?
dataset = {
    "The idea of a 'community of voices' sounds as though it would work on paper but is rather imperfect when you delve into the different implications it can hold for the community and just what it means to have one. While it could be that we should have something such as a community of voices wherein opinions can be expressed as a whole, one must take a closer look at what that really means. To have a community of voices, that would mean many people whose backgrounds can vary." : ["unclear", "vague", "nonsensical", "cognition", "confusing", "argue"],
    "th United states didnt listen to the small protests thruout the 19th centuri regarding peopels rights until the 20th centry when voices of many drowned out the few. Posters dont matter till their held by millions. And only when we r more involved then not does Congress here and respond. But evenchally with the voice of millons behind the movement the gov listen and change. The millon voices sad black isnt bad and got world to chang ebcause of it.": ["Spelling", "grammar", "logical", "sentence", "structure"]
}

# print all supported models in promptbench
print('All supported models: ')
print(pb.SUPPORTED_MODELS)

# load a model, flan-t5-large, for instance.
model = pb.LLMModel(model='gpt-3.5-turbo', max_new_tokens=300, temperature=0.0001)
#model = pb.LLMModel(model='llama2-13b-chat', max_new_tokens=10, temperature=0.0001)

# Prompt API supports a list, so you can pass multiple prompts at once.

prompts = pb.Prompt([
    "You are a doctoral English student with a focus on pedagogical studies giving feedback on a high school student's AP Language exam essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”.: ", 
    "You are giving feedback on a student's essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”. For example, does the student use proper grammar, spelling, and argument structure?: ",
    "You are a doctoral English student with a focus on pedagogical studies giving feedback on a high school student’s AP Language exam essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”.",
                     ])

#data -- with promptbench, in this example specifically it looks like the data is labelled with a yes or no if it's positive or negative
def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred, -1)

def has_any(reply, keywords):
    return any(substring in reply for substring in keywords)

#1 if response contains ideal keywords, 0 if doesn't contain keywords
#CONSIDER -- ideal NUMBER of keywords in response???
hasKeywords = 0
total = 0
for prompt in prompts:
    preds = []
    labels = []
    for essay_sample, keywords in tqdm(dataset.items()):
        # process input
        input_text = prompt + " " + essay_sample #pb.InputProcess.basic_format(prompt, essay_sample)
        print("input text: ", input_text)
        #label = inputString['label']
        raw_pred = model(input_text)
        print("output: ", raw_pred)
        print()
        preds.append(raw_pred)
        if(has_any(raw_pred, keywords)):
            hasKeywords += 1
        total += 1
        
    # evaluate
    #score = pb.Eval.compute_cls_accuracy(preds, labels)
    #print(f"{score:.3f}, {prompt}")

print("RUNNED")