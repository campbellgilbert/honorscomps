from openai import OpenAI
import promptbench as pb
import numpy as np
#idea here is to either read in an excel file or just take in data and output the proper .json file
#in the openai includes eval style
#we want to create the json dataset; then, in a different file, run the set

"""response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]

  create_json
)
print(response.choices[0].message.content)"""

"""
for each (systemPrompt in systemPrompts)
    {role: "system", "content": "....."}
    {role: "user", "content": ...}
    create new json string file
"""

#LIST OF PROMPTS

prompts = np.array(
    ["You are a high school English teacher giving feedback on a student's AP Language essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”",
    "You are giving feedback on a student's essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”. Pay close attention to the thesis: does it only restate the prompt, take a vague or unclear position, or state an obvious fact rather than making a defensible claim? If the section is from a body paragraph with no thesis, does it appear to be supporting a lucid argument?",
    "You are a gifted high school AP English student giving feedback on a peer's AP Language exam. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”.",
    "You are giving feedback on a student's essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”. Pay close attention to the evidence and commentary: is the evidence incoherent, overly general, or irrelevant to the argument? Does the commentary fail to explain how the evidence supports the line of reasoning, and is there a logical, non-faulty line of reasoning to begin with? If the section is from the thesis paragraph, is it setting up a well-defensible argument?",
    "You are a college English professor giving feedback on a high school student's AP Language exam essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”. ",
    "You are giving feedback on a student's essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”. Pay close attention to argument sophistication and cognition: is the argument un-nuanced, ignoring complexities and tensions? Are implications and limitations articulated within a broader context? Are the rhetorical choices ineffective and is the style vivid and persuasive enough?",
    "You are an AP exam grader giving feedback on a high school student's AP Language exam essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”.",
    "You are giving feedback on a student's essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”. For example, does the student respond to the prompt with a thesis that presents a defensible position? Is evidence provided to support the line of reasoning, and is that evidence explained and articulated upon?",
    "You are a doctoral English student with a focus on pedagogical studies giving feedback on a high school student's AP Language exam essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”.",
    "You are giving feedback on a student's essay. Point out any issues with the following writing sample; if there is nothing noticeably missing or incorrect, say “Looks good!”. FOr example, does the student use proper grammar, spelling, and argument structure?"]
)

samples_and_keywords = np.array(
    [
        ("The United States did not listen to the small protests throughout the nineteenth century regarding people's rights. It wasn't until the twentieth century when millions of Americans took to the streets to fight against racial inequality. And only then when more were involved than not, did Congress hear and respond. All this after an entire Civil War. Eventually, millions of voices joined a movement.", 
         "commentary, elaborate, contextualize"),
        ("The United States did not listen to the small protests throughout the nineteenth century regarding people's rights. It wasn't until the twentieth century when the voices of many drowned out the voices of few within Washington. Posters didn't matter until held by millions. Black was bad until millions said it wasn't. And only then when more were involved than not, did Congress hear and respond. It took an entire Civil War to see the problem the first time and it took march after march, protest after protest to bring more change in the 1960s. But eventually, with the voice of millions behind the movement, the government listened and changed. The million voices said black isn't bad, something people now realize is true, and got the world to change because of it.", 
         "Looks good!"),
        ("The United States didn't listen to the small protests throughout the nineteenth century. It wasn't until the twentieth century when the voices of many drowned out the voices of few within Washington. Posters didn't matter until held by millions. And only then when more were involved than not, did Congress hear and respond. It took an entire war to see a problem the first time and it took march after march, protest after protest to bring more change. But eventually, with the voice of millions behind the movement, the government listened and changed.",
         "Evidence, clarity, precise, lucid, specific, general"),
        ("th United states didnt listen to the small protests thruout the 19th centuri regarding peopels rights until the 20th centry when voices of many drowned out the few. Posters dont matter till their held by millions. And only when we r more involved then not does Congress here and respond. But evenchally with the voice of millons behind the movement the gov listen and change. The millon voices sad black isnt bad and got world to chang ebcause of it.", "spelling, grammar, logical, sentence, structure")
        ("What will people hear? A whisper or a song? The voice of the one or the voice of the many, united? One alone can mean very little until that one voice gains the power of many. Humans are creatures that crave validation and to fit in, so will they be one alone or be one of many? Will a person be Moses or the Jews? A community of voices can hold more power than one voice alone, but one voice can be used to inspire many, as shown throughout history through civil rights protests, Hitler, and the United States.", "Looks good!"),
        ("What will people hear? A whisper or a song? The voice of the one or the voice of the many, united? Humans are creatures that crave validation and to fit in, so will they be one alone or be one of many? Will a person be Moses or the Jews?", "cognition, reasoning, precise, develops, disputable, thesis"),
        ("What will people hear? A whisper or a song? The voice of the one or the voice of the many, united? One alone can mean very little until that one voice gains the power of many. Humans are creatures that crave validation and to fit in, so will they be one alone or be one of many? A community of voices can hold more power than one voice alone, but one voice can be used to inspire many.", "Evidence, specific, elaborate, vague"),

        ("The idea of a 'community of voices' sound as though it would work on paper but is rather imperfect when you delve into the different implications it can hold for the community and just what it means to have a.While yes it's a very nice idea that we should have something such as a community of voices where peoples opinions can be expressed as a whole, you have to take a closer look at what that really means. To have a community of voices, that would mean many people whose back-rounds can vary.", "Thesis, point, vague, grammar, syntax, spelling, clear"),

        ("The idea of a 'community of voices' sounds as though it would work on paper but is rather imperfect when you delve into the different implications it can hold for the community and just what it means to have one. While it could be that we should have something such as a community of voices wherein opinions can be expressed as a whole, one must take a closer look at what that really means. To have a community of voices, that would mean many people whose backgrounds can vary.", "Unclear, vague, nonsensical, cognition, confusing, argue")
    ],
    dtype=[('writing sample', 'U1000'), ('keywords', 'U100')]
)
#gonna iterate thru prompts, attach each example of sample-to-keyword to prompt



#LIST OF WRITING SAMPLE:KEYWORD PAIRS

"""Each JSON object will represent one data point in your eval. The keys you need in the JSON object depend on the eval template. 
All templates expect an "input" key, which is the prompt, ideally specified in chat format (though strings are also supported). 
We recommend chat format even if you are evaluating non-chat models. 
If you are evaluating both chat and non-chat models, we handle the conversion between chat-formatted prompts and raw string prompts (see the conversion logic here).

For the basic evals Match, Includes, and FuzzyMatch, the other required key is "ideal", which is a string (or a list of strings) specifying the correct reference answer(s). 
For model-graded evals, the required keys vary based on the eval but is determined by the {key}s in the evaluation prompt that are not covered by the (optional) args.

"""

"""
{"input": 
    [
        {"role": "system", 
        "content": "You are now an emotionally intelligent AI. In this test, you will be presented with a few brief details about an emotional situation, and asked to choose from among four responses what you would be most likely to do in the situation. Please note, there are no right or wrong answers. We all deal with situations in different ways. All that you need to do is answer each question honestly. Select one or more response(s) by returning the one or more corresponding lowercase letter(s) ('a', 'b', 'c', or 'd'), and, if you selected more than one, sorting them, separated by hyphen, in the order that you think best ranks them from most to least effective, within the context of the vignette provided."}, 
        {"role": "user", 
        "content": "Your best friend is moving to another state and is unlikely to come back. You have been good friends for many years. What would you do in this situation? (a) Make sure you both keep in contact through email, phone, or letter writing; (b) Spend time with other friends, and keep busy; (c) Hope that your best friend will return soon; (d) Forget about your best friend"}
    ], 
    "ideal": "b-c-a-d"}"""

"""
{"input": 
    [
        {"role": "system", 
        "content": "You are giving feedback on a student's essay. Point out any issues with the following writing sample, which is a section of the essay; if there is nothing noticeably missing or incorrect, say “Looks good!. The user is responding to the following prompt: 
        “In a 2016 interview published in the Los Angeles Review of Books, Maxine Hong Kingston, an award-winning writer famous for her novels depicting the experiences of Chinese immigrants in the United States, stated: 'I think that individual voices are not as strong as a community of voices. If we can make a community of voices, then we can speak more truth.' Write an essay that argues your position on the extent to which Kingston's claim about the importance of creating a community of voices is valid."}, 

        {"role": "user", 
        "The United States did not listen to the small protests throughout the nineteenth century regarding people's rights. It wasn't until the twentieth century when millions of Americans took to the streets to fight against racial inequality. And only then when more were involved than not, did Congress hear and respond. All this after an entire Civil War. Eventually, millions of voices joined a movement.

"}
    ], 
    "ideal": "commentary, elaborate, contextualize"}"""