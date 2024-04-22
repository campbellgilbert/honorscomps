import promptbench as pb
import json
#from eval import Evaluator
#from model import ModelWrapper

# Initialize the dataset
grade = open("GRADE.json")
gradeData = json.load(grade)
print(gradeData)

dataset = pb.DatasetLoader.load_dataset("sst2")

# Initialize the model
model = ModelWrapper(model_name='openai/gpt-3')  # Adjust as needed for your specific model

# Initialize the evaluator with the dataset and model
evaluator = Evaluator(model=model, dataset=dataset)

# Define the prompts for evaluation
prompts = [
    "You are a high school English teacher giving feedback on a student’s AP Language essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”.",
    "You are a gifted high school AP English student giving feedback on a peer’s AP Language exam. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”."
]

# Function to run evaluation for each prompt
def run_evaluation():
    results = {}
    for i, prompt in enumerate(prompts, 1):
        print(f"Evaluating with prompt {i}: {prompt}")
        evaluation_result = evaluator.evaluate(prompt=prompt)
        results[f'Prompt {i}'] = evaluation_result
        print(f"Results for Prompt {i}: {evaluation_result}")
    return results

# Execute the evaluation
final_results = run_evaluation()
"""