from openai import OpenAI
import os
import promptbench as pb
import csv
# my data rows as dictionary objects
mydict = [{'branch': 'COE', 'cgpa': '9.0',
           'name': 'Nikhil', 'year': '2'},
          {'branch': 'COE', 'cgpa': '9.1',
           'name': 'Sanchit', 'year': '2'},
          {'branch': 'IT', 'cgpa': '9.3',
           'name': 'Aditya', 'year': '2'},
          {'branch': 'SE', 'cgpa': '9.5',
           'name': 'Sagar', 'year': '1'},
          {'branch': 'MCE', 'cgpa': '7.8',
           'name': 'Prateek', 'year': '3'},
          {'branch': 'EP', 'cgpa': '9.1',
           'name': 'Sahil', 'year': '2'}]
 
# field names
fields = ['name', 'branch', 'year', 'cgpa']
 
# name of csv file
filename = "data.csv"
 
# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv dict writer object
    writer = csv.DictWriter(csvfile, fieldnames=fields)
 
    # writing headers (field names)
    writer.writeheader()
 
    # writing data rows
    writer.writerows(mydict)

"""
dataset_name = "gsm8k"
dataset = pb.DatasetLoader.load_dataset(dataset_name)

method = pb.PEMethod(method='emotion_prompt', 
                        dataset=dataset_name,
                        verbose=True,  # if True, print the detailed prompt and response
                        prompt_id = 1  # for emotion_prompt 
                        )

print('All supported models: ')
print(pb.SUPPORTED_MODELS)

# load a model, flan-t5-large, for instance.
model = pb.LLMModel(model='gpt-3.5-turbo', max_new_tokens=300, temperature=0.0001)

results = method.test(dataset, 
                      model, 
                      num_samples=5 # if don't set the num_samples, method will use all examples in the dataset
                      )
"""