import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

def extract_keywords(text, totallen):
    #step 1: clean up text
    translator = str.maketrans('', '', string.punctuation)
    text = str(text).translate(translator)
    #step 2: tokenize
    words = word_tokenize(text.lower())
    #step 3: update averages
    totallen += len(words)
    stop_words = set(stopwords.words('english'))
    #step 4: clean up and return keywords
    keywords = [word for word in words if word not in stop_words]
    return keywords, totallen

#load in excel file -- specifically just the outputs

#this code by claude
# Load the Excel files into pandas DataFrames
df1 = pd.read_excel('database.xlsx')
df2 = pd.read_excel('database_claude.xlsx')

# Select the 8th column (index 7) from both DataFrames
gptoutputs = df1.iloc[:, 5]
claudeoutputs = df2.iloc[:, 5]

chat_words_total = 0
claude_words_total = 0

# Iterate over the cells in both columns
for cell1, cell2 in zip(gptoutputs, claudeoutputs):
    # Extract keywords from each cell
    
    keywords_chat, chat_words_total = extract_keywords(cell1, chat_words_total)
    keywords_claude, claude_words_total = extract_keywords(cell2, claude_words_total)
    
    # Find common keywords between the two cells
    common_keywords = set(keywords_chat) & set(keywords_claude)
    
    # Print the common keywords
    if common_keywords:
        print(f"Common keywords: {', '.join(common_keywords)}")
    else:
        print(f"No common keywords found")
    print()


#find some stuff to get back

"""
output length, average response, chat
output length, average response, claude

top 5 words, chat
top 5 words, claude

most DISSIMILAR promp


"""
print("Avg ChatGPT output len: ", (chat_words_total / 211))
print("Avg Claude output len: ", (claude_words_total / 211))

print("Most common words, ChatGPT: ")
print("Most common words, Claude: ")
print("Most common words, overall: ")

print("Mr Struggle, overall:")
print("Mr Successful, overall:")




