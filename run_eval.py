from openai import OpenAI
import os
api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI()

prompts = [
    "You are a doctoral English student with a focus on pedagogical studies giving feedback on a high school student's AP Language exam essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”.",
    "You are giving feedback on a student's essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”. For example, does the student use proper grammar, spelling, and argument structure?",
    "You are a doctoral English student with a focus on pedagogical studies giving feedback on a high school student’s AP Language exam essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”."
]

#dataset dictionary
dataset = {
    "The idea of a 'community of voices' sounds as though it would work on paper but is rather imperfect when you delve into the different implications it can hold for the community and just what it means to have one. While it could be that we should have something such as a community of voices wherein opinions can be expressed as a whole, one must take a closer look at what that really means. To have a community of voices, that would mean many people whose backgrounds can vary." : ["unclear", "vague", "nonsensical", "cognition", "confusing", "argue"],
    "th United states didnt listen to the small protests thruout the 19th centuri regarding peopels rights until the 20th centry when voices of many drowned out the few. Posters dont matter till their held by millions. And only when we r more involved then not does Congress here and respond. But evenchally with the voice of millons behind the movement the gov listen and change. The millon voices sad black isnt bad and got world to chang ebcause of it.": ["Spelling", "grammar", "logical", "sentence", "structure"]
}

responses = []
correct = 0
total = 0
for prompt in prompts:
    for essay_sample, keywords in dataset.items():
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": essay_sample}]
        )
        reply = response.choices[0].message.content
        print("question: ", essay_sample)
        print()
        print("response: ", reply)
        responses.append(reply)
        if(any(substring in reply for substring in keywords)):
            
            correct += 1
        total += 1

print("total: ", total)
print("correct: ", correct)