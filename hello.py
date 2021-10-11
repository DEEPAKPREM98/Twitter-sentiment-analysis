import pickle
from keras.models import load_model
import string
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sys
import text2emotion as te
import tweepy as tw
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Embedding, GRU, LSTM, Bidirectional
from keras.models import Sequential
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
import re
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

model = load_model('model.h5')

consumerKey = "WIw5NQD458LlSPx7Fsvw0eg9m"
consumerSecret = "vX68sroX0n8MNhzRBFMHJcoaY0tEZj1hp5MjvLIDFaeLHthyfb"
accessToken = "947884897720344576-kuHtcOSVjiju4q8EZ5kqfuwGf2rEN30"
accessTokenSecret = "RGLm5qJcGg3Hpie4wkgkueQ33T8DpsOlu1fFwy3blSR9l"

# Create the authentication object
authenticate = tw.OAuthHandler(consumerKey, consumerSecret)

# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)

# Creating the API object while passing in auth information
api = tw.API(authenticate, wait_on_rate_limit=True)

comp_searches = []

# Define the search term and the date_since date as variables
f = open("put.txt", "r")
t = f.read()

for i in range(1, int(t)-1):
    comp_searches.append(sys.argv[i])

date_since = "2020-11-16"

users_locs = []
newlist = []
for search in comp_searches:
    new_search = search + " -filter:retweets"
    tweets = tw.Cursor(api.search, q=new_search, lang="en",
                       since=date_since).items(int(sys.argv[int(t)-1]))
    users_locs = [[tweet.created_at, search, tweet.user.screen_name,
                   tweet.text] for tweet in tweets]
    newlist = newlist+users_locs
    del users_locs[:]


df = pd.DataFrame(data=newlist,
                  columns=["Date", "search_word", "User", "Tweets", ])


def clean_text(text):
    text = text.lower()

    pattern = re.compile(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    text = pattern.sub('', text)
    text = " ".join(filter(lambda x: x[0] != '@', text.split()))
    emoji = re.compile("["
                       u"\U0001F600-\U0001FFFF"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)

    text = emoji.sub(r'', text)
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"did't", "did not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"couldn't", "could not", text)
    text = re.sub(r"have't", "have not", text)
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text


def CleanTokenize(df):
    tweet = list()
    lines = df["Tweets"].values.tolist()

    for line in lines:
        line = clean_text(line)
        # tokenize the text
        tokens = word_tokenize(line)
        # remove puntuations
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove non alphabetic characters
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words("english"))
        # remove stop words
        words = [w for w in words if not w in stop_words]
        tweet.append(words)
    return tweet


tweet = CleanTokenize(df)

max_length = 25

tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(tweet)

# retrieve:
f = open('history.pckl', 'rb')
history = pickle.load(f)
f.close()

print('Summary of the built model...')
print(model.summary())


def predict_sarcasm(s):
    x_final = pd.DataFrame({"Tweets": [s]})
    test_lines = CleanTokenize(x_final)
    test_sequences = tokenizer_obj.texts_to_sequences(test_lines)
    test_review_pad = pad_sequences(
        test_sequences, maxlen=max_length, padding='post')
    pred = model.predict(test_review_pad)
    pred *= 100
    if pred[0][0] >= 99:
        return "sarcastic"
    else:
        return "not_sarcastic"


plt.style.use('fivethirtyeight')

df['Tweets'] = df['Tweets'].apply(clean_text)
df['Sarcastic'] = df['Tweets'].apply(predict_sarcasm)

scores = []

for i in range(df['Tweets'].shape[0]):
    scores.append(analyser.polarity_scores(df['Tweets'][i]))

gf = pd.DataFrame(scores)
df = df.join(gf)

del df["neg"]
del df["neu"]
del df["pos"]

dore = []

for i in range(df['Tweets'].shape[0]):
    dore.append(te.get_emotion(df['Tweets'][i]))

hf = pd.DataFrame(dore)
df = df.join(hf)


def getAnalysis(core):
    if core <= -0.05:
        return 'Negative'
    elif core >= 0.05:
        return 'Positive'
    else:
        return 'Neutral'


df["Sentiment"] = df['compound'].apply(getAnalysis)

print(df)

df.to_csv('static\ile1.csv')


def perc(x):
    y = (x*100)/int(sys.argv[int(t)-1])
    return(y)


#compound plot
score_tabl = df.pivot_table(
    index='search_word',  values="compound", aggfunc=np.mean)
score_tabl.to_csv('static\ile2.csv')
dat = pd.read_csv('static\ile2.csv')
dp = pd.DataFrame(dat)
plt.bar(dp['search_word'], dp['compound'])
plt.xlabel("search-word")
plt.ylabel("compound score")
plt.legend()
plt.tight_layout()
plt.savefig("static\p1.jpg")


#sarcastic plot
li = []
lis = []
for i in comp_searches:
    df1 = df[df['search_word'] == i]
    x = df1['Sarcastic'].value_counts()
    li.append(x.sarcastic)
    lis.append(x.not_sarcastic)

dy = pd.DataFrame()
dy['search_word'] = comp_searches
dy['sarcastic'] = li
dy['not_sarcastic'] = lis
dy["sarcastic %"] = dy['sarcastic'].apply(perc)
dy["non_sarcastic %"] = dy['not_sarcastic'].apply(perc)
dy.to_csv('static\ile5.csv')
print((dy))

del dy["sarcastic %"]
del dy["non_sarcastic %"]

z = comp_searches
z_axis = np.arange(len(z))
dy.plot.bar()
plt.ylabel("No of tweets")
plt.xticks(z_axis, z)
plt.legend(prop={"size": 8})
plt.tight_layout()
plt.savefig("static\p2.jpg")

#emotion plot
score_table = df.pivot_table(
    index='search_word',  values=("Happy", "Angry", "Surprise", "Sad", "Fear"), aggfunc=np.mean)
print((score_table))
score_table.to_csv('static\ile3.csv')
data = pd.read_csv('static\ile3.csv')
du = pd.DataFrame(data)
cos = du["search_word"].tolist()
X = cos
X_axis = np.arange(len(X))
du.plot.bar()
plt.ylabel("Emotion score")
plt.xticks(X_axis, X)
plt.legend(prop={"size": 8})
plt.tight_layout()
plt.savefig("static\p3.jpg")


#sentiment plot
liv = []
lisv = []
livs = []
for i in comp_searches:
    df1 = df[df['search_word'] == i]
    x = df1['Sentiment'].value_counts()
    liv.append(x.Positive)
    lisv.append(x.Negative)
    livs.append(x.Neutral)

dyi = pd.DataFrame()
dyi['search_word'] = comp_searches
dyi['Positive'] = liv
dyi['Negative'] = lisv
dyi['Neutral'] = livs
dyi['Positive %'] = dyi['Positive'].apply(perc)
dyi['Negative %'] = dyi['Negative'].apply(perc)
dyi['Neutral %'] =  dyi['Neutral'].apply(perc)
dyi.to_csv('static\ile4.csv')
print((dyi))

del dyi['Positive %']
del dyi['Negative %']
del dyi['Neutral %']


z = comp_searches
z_axis = np.arange(len(z))
dyi.plot.bar()
plt.ylabel("No of tweets")
plt.xticks(z_axis, z)
plt.legend(prop={"size": 8})
plt.tight_layout()
plt.savefig("static\p4.jpg")
