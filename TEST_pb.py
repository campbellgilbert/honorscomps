
import promptbench as pb
from tqdm import tqdm

#client = OpenAI("sk-IN1fbBq1GLvQTYQrymjrT3BlbkFJcuvCrbj6e8BezNGZW3mF3")

# print all supported datasets in promptbench
#print('All supported datasets: ')
#print(pb.SUPPORTED_DATASETS)
dataset = pb.DatasetLoader.load_dataset("mmmu")
model = pb.VLMModel(model='llava-hf/llava-1.5-7b-hf', max_new_tokens=2048, temperature=0.0001, device='cuda')
prompts = pb.Prompt([
    "You are a helpful assistant. Here is the question:{question}\nANSWER:",
    "USER:{question}\nANSWER:",
])

for prompt in prompts:
    preds = []
    labels = []
    for data in tqdm(dataset):
        # process input
        input_text = pb.InputProcess.basic_format(prompt, data)
        input_images = data['images']  # please use data['image_paths'] instead of data['images'] for models that only support image path/url, such as GPT-4v
        label = data['answer']
        raw_pred = model(input_images, input_text)
        # process output
        pred = pb.OutputProcess.pattern_split(raw_pred, 'ANSWER:')
        preds.append(pred)
        labels.append(label)
    
    # evaluate
    score = pb.Eval.compute_cls_accuracy(preds, labels)
    print(f"{score:.3f}, {repr(prompt)}")