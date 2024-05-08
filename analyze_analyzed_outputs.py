import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Load the specified Excel file
data = pd.read_excel('feedback_analysis_2.xlsx')

#avg expected keywords
avg_expec
#avg expanded expected keywords

#avg semantically-meaningful keywords

#avg f-k score

avg_response_length_claude = merged_df['6. output_claude'].str.len().mean()
avg_response_length_chatgpt = merged_df['6. output_chatgpt'].str.len().mean()


def unique_words(text):
    # Split the text into words, convert to lowercase to ensure uniqueness is case-insensitive
    words = text.lower().split()
    # Use a set to keep only unique words, then return the length of the set
    return len(set(words))

expected_keywords_claude = data.iloc[:, 8].dropna()
expected_keywords_chatgpt = data.iloc[:, 9].dropna()

similar_keywords_claude = data.iloc[:, 10].dropna()
similar_keywords_chatgpt = data.iloc[:, 11].dropna()



fk_claude = data.iloc[:, 14].dropna()
fk_chat = data.iloc[:, 15].dropna()




outputs = []
for i in [expected_keywords_claude, similar_keywords_claude, expected_keywords_chatgpt, similar_keywords_chatgpt]:
    unique_counts = [unique_words(text) for text in i if text] 
    unique_avg = sum(unique_counts) / len(unique_counts)
    
    outputs.append(unique_avg)

outputs.append(sum(fk_claude) / len(fk_claude))
outputs.append(sum(fk_chat) / len(fk_chat))


i = 0
outputpt1 = ["expected keywords", "similar keywords", "f-k score"]
outputpt2 = ["claude", "gpt-4"]
for m in outputpt1:
    for n in outputpt2:
        print("average", m, "for", n, ": ", outputs[i])
        i += 1
         
#average expected keywords (unique), claude
unique_counts_exp_claude = [unique_words(text) for text in expected_keywords_claude if text]  
unique_counts_avg_claude = sum(unique_counts_exp_claude) / len(unique_counts_exp_claude)

print("avg unique expected words, claude: ", unique_counts_avg_claude)

#average expected keywords (unique), gpt-4


#average similar-to-expected keywords (unique), claude

#average similar-to-expected keywords (unique), gpt-4


#average F-K, claude

#average F-K, gpt-4