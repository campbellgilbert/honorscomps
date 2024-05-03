import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def extract_keywords(text, totallen):
    words = word_tokenize(str(text).lower())
    #update avgs :33
    totallen += len(words)

    stop_words = set(stopwords.words('english'))
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
        print(f"Common keywords between '{cell1}' and '{cell2}': {', '.join(common_keywords)}")
    else:
        print(f"No common keywords found between '{cell1}' and '{cell2}'")

print("Avg ChatGPT output len: ", (chat_words_total / 211))
print("Avg Claude output len: ", (claude_words_total / 211))