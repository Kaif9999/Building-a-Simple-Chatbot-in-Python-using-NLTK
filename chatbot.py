# import necessary libraries
import io
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import nltk
from nltk.stem import WordNetLemmatizer

# download nltk packages if not already downloaded
nltk.download('popular', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# reading in the corpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

# tokenization
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

# preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# keyword matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

# initialize conversation history
conversation_history = []

print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")
while True:
    user_response = input()
    user_response=user_response.lower()

    # check for exit condition
    if user_response == 'bye':
        print("ROBO: Bye! take care..")
        break

    # check for thanks
    elif user_response in ['thanks', 'thank you']:
        print("ROBO: You are welcome..")

    # check for greeting
    elif greeting(user_response) is not None:
        print("ROBO: "+greeting(user_response))

    # for any other input, generate a response
    else:
        robo_response = response(user_response)
        print("ROBO: " + robo_response)
        conversation_history.append("USER: " + user_response)
        conversation_history.append("ROBO: " + robo_response)

# Adding conversation to history
with open('conversation_history.txt', 'a') as f:
    for line in conversation_history:
        f.write(line + "\n")
