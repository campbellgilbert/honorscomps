import promptbench as pb
from tqdm import tqdm

# print all supported datasets in promptbench
print('All supported datasets: ')
print(pb.SUPPORTED_DATASETS)

# load a dataset, sst2, for instance.
# if the dataset is not available locally, it will be downloaded automatically.
dataset = pb.DatasetLoader.load_dataset("mmlu")
# dataset = pb.DatasetLoader.load_dataset("mmlu")
# dataset = pb.DatasetLoader.load_dataset("un_multi")
# dataset = pb.DatasetLoader.load_dataset("iwslt2017", ["ar-en", "de-en", "en-ar"])
# dataset = pb.DatasetLoader.load_dataset("math", "algebra__linear_1d")
# dataset = pb.DatasetLoader.load_dataset("bool_logic")
# dataset = pb.DatasetLoader.load_dataset("valid_parenthesesss")
print()
# print the first 5 examples
print(dataset[:5])
print()
# print all supported models in promptbench
print('All supported models: ')
print(pb.SUPPORTED_MODELS)

"""
# load a model, flan-t5-large, for instance.
model = pb.LLMModel(model='gpt-4', max_new_tokens=10, temperature=0.0001, device='cuda')
# model = pb.LLMModel(model='llama2-13b-chat', max_new_tokens=10, temperature=0.0001)

prompts = pb.Prompt(["Classify the sentence as positive or negative: {content}",
                     "Determine the emotion of the following sentence as positive or negative: {content}"
                     ])

def proj_func(pred):
    mapping = {
        "positive": 1,
        "negative": 0
    }
    return mapping.get(pred, -1)

for prompt in prompts:
    preds = []
    labels = []
    for data in tqdm(dataset):
        # process input
        input_text = pb.InputProcess.basic_format(prompt, data)
        label = data['label']
        raw_pred = model(input_text)
        # process output
        pred = pb.OutputProcess.cls(raw_pred, proj_func)
        preds.append(pred)
        labels.append(label)
    
    # evaluate
    score = pb.Eval.compute_cls_accuracy(preds, labels)
    print(f"{score:.3f}, {prompt}")"""