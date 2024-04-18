from openai import OpenAI
#WE NOW HAVE THIS RUNNING!!!!!!!!!!!!!!!!!!!11  
#sk-IN1fbBq1GLvQTYQrymjrT3BlbkFJcuvCrbj6e8BezNGZW3mF3
#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 <- use this one
import os
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)

## Use GPT-4 to generate synthetic data
# Define the system prompt and user input (these should be filled as per the specific use case)
