# import numpy as np
# import tensorflow as tf
# import tensorflow_hub as hub
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import string
# # nltk.download('stopwords')
# import pandas as pd
#
# # nltk.download('punkt')
# # word_list = [
# #     'I', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
# #     'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
# #     'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
# #     'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
# #     'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
# #     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
# #     'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
# #     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
# #     'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
# #     'through', 'during', 'before', 'after', 'above', 'below', 'to',
# #     'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
# #     'again', 'further', 'then', 'once', 'here', 'there', 'when',
# #     'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
# #     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
# #     'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
# #     'don', 'should', 'now'
# # ]
#
# df = pd.read_csv('dataset-final.csv')
#
# # Get the text column
# disorders = ['Mood Issue', 'Sleep Issue', 'Anxiety Issue', 'Depression', 'Eating Disorder', 'Panic Disorder',
#              'Obsessive-Compulsive', 'Hallucinations', 'Self-Harm', 'Bipolar Disorder', 'PTSD', 'Suicidal Thoughts']
# table = df.values
# texts = df['Text Prompt'].values
# Sample dataset containing text
#
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
#
#
# def get_text_embeddings(texts):
#     embeddings = embed(texts)
#     return embeddings
#
#
# def preprocess_text(text):
#     # Convert to lowercase
#     text = text.lower()
#
#     # Remove punctuation
#     # text = ''.join([char for char in text if char not in string.punctuation])
#
#     # Tokenize the text
#     tokens = word_tokenize(text)
#
#     # Remove stopwords
#     tokens = [word for word in tokens if word not in texts]
#
#     # Join tokens back into a string
#     text = ' '.join(tokens)
#
#     return text
#
#
# df['Text Prompt'] = df['Text Prompt'].apply(preprocess_text)
#
# embeddings = get_text_embeddings(texts)
#
# # Input text for which you want to find similar texts
# input_text = preprocess_text("i am feeling sad. It hurts to try and be happy. I don't feel anything")
#
#
# def botResponse(prompt):
#     input_text = preprocess_text(prompt)
#     # Compute the embedding for the input text
#     input_embedding = get_text_embeddings([input_text])[0]
#
#     # Compute cosine similarities between the input text and dataset
#     cosine_similarities = cosine_similarity([input_embedding], embeddings)
#
#     # Get the top 5 most similar texts
#     top_indices = cosine_similarities.argsort()[0][-5:][::-1]
#
#     # Print the most similar texts
#     print("Input Text:")
#     print(input_text)
#     print("\nTop 5 Similar Texts:")
#     resList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     explain = [0, 0, 0, 0, 0]
#     for i, idx in enumerate(top_indices):
#         for disorder in range(1, 13):
#             resList[disorder] += table[idx][disorder]
#             explain[i] = table[idx][13]
#     resList = resList[1:]
#     resDisorders = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
#     ans = {"Disorders": [], "Percentage": [], "Explanation": []}
#     total = sum(resList)
#     for dNum in range(len(resList) - 1):
#         resDisorders[dNum] = (int(resList[dNum]) / total) * 100
#     resList = resDisorders[1:]
#     for i in range(len(resList)):
#         if resDisorders[i] != 0:
#             ans["Disorders"].append(disorders[i])
#             ans["Percentage"].append(int(resDisorders[i]))
#     ans["Explanation"] += (explain)
#     return ans
#
#
# botResponse("I want to hurt myself and suicidal")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
# nltk.download('stopwords')
import pandas as pd

# nltk.download('punkt')
# word_list = [
#     'I', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
#     'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
#     'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
#     'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what',
#     'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am',
#     'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
#     'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
#     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
#     'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
#     'through', 'during', 'before', 'after', 'above', 'below', 'to',
#     'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
#     'again', 'further', 'then', 'once', 'here', 'there', 'when',
#     'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
#     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
#     'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
#     'don', 'should', 'now'
# ]

df = pd.read_csv('dataset-final.csv')
disorders = ['Mood Issue', 'Sleep Issue', 'Anxiety Issue', 'Depression', 'Eating Disorder', 'Panic Disorder',
             'Obsessive-Compulsive', 'Hallucinations', 'Self-Harm', 'Bipolar Disorder', 'PTSD', 'Suicidal Thoughts']
table = df.values
texts = df['Text Prompt'].values

# Get the text column
texts = df['Text Prompt'].values
# Sample dataset containing text

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def get_text_embeddings(texts):
    embeddings = embed(texts)
    return embeddings


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    # text = ''.join([char for char in text if char not in string.punctuation])

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in texts]

    # Join tokens back into a string
    text = ' '.join(tokens)

    return text


df['Text Prompt'] = df['Text Prompt'].apply(preprocess_text)

embeddings = get_text_embeddings(texts)
def botResponse(prompt):
    input_text = preprocess_text(prompt)
    # Compute the embedding for the input text
    input_embedding = get_text_embeddings([input_text])[0]

    # Compute cosine similarities between the input text and dataset
    cosine_similarities = cosine_similarity([input_embedding], embeddings)

    # Get the top 5 most similar texts
    top_indices = cosine_similarities.argsort()[0][-5:][::-1]

    # Print the most similar texts
    print("Input Text:")
    print(input_text)
    print("\nTop 5 Similar Texts:")
    resList = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    explain = [0, 0, 0, 0, 0]
    for i, idx in enumerate(top_indices):
        for disorder in range(1, 13):
            resList[disorder] += table[idx][disorder]
            explain[i] = table[idx][13]
    resList = resList[1:]
    resDisorders = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ans = {"Disorders": [], "Percentage": [], "Explanation": []}
    total = sum(resList)
    for dNum in range(len(resList) - 1):
        resDisorders[dNum] = (int(resList[dNum]) / total) * 100
    resList = resDisorders[1:]
    print(resDisorders)
    for i in range(len(resList)):
        if resDisorders[i] != 0:
            ans["Disorders"].append(disorders[i])
            ans["Percentage"].append(int(resDisorders[i]))
    ans["Explanation"] += (explain)
    return ans

@app.get("/chat")
async def chat(query: str):
    return botResponse(query)


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(app, port=int(os.environ.get('PORT', 8081)), host="127.0.0.1")
