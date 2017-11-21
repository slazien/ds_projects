import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
from tqdm import tqdm

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


def clean_tweets(tweets):
    # takes a list of tweets and outputs cleaned tweets list

    cleaned_tweets = []

    for tweet in tweets:
        tweet_temp = " ".join(tweet.split())  # strip duplicate whitespace
        tweet_temp = re.sub("\.{4,}", "...", tweet_temp)  # replace multiple ..... with ...
        tweet_temp = re.sub(r"^https?://.*[\r\n]*", "", tweet_temp)  # strip URLs
        tweet_temp = re.sub(r"[@#$%^&*~]", "", tweet_temp)  # remove some weird characters
        tweet_temp = tweet_temp.lower()  # to lowercase
        """ replace 3+ consecutive occurrences of a char by "n" chars to preserve 'excitement' (watch out with that!)
        since it might change the vocabulary for words such as 'too' etc."""
        matches = re.findall(r"((\w)\2{2,})", tweet_temp)
        n = 2
        if matches is not None:
            for match in matches:
                tweet_temp = re.sub(match[0], match[1]*n, tweet_temp)
        cleaned_tweets.append(tweet_temp)

    return cleaned_tweets


def tokenize_tweets(tweets, n_cores=8):
    # Tokenize tweets using multiple CPU cores if needed

    tokenized_tweets = []

    bar = tqdm(total=len(cleaned_tweets))
    pool = Pool(processes=n_cores, maxtasksperchild=1)
    for text in pool.imap(word_tokenize, tweets, chunksize=10000):
        tokenized_tweets.append(text)
        bar.update()

    return tokenized_tweets


def get_words(tweets):
    # Concatenate all tweets into one huge list of words (needed for word freq counts)

    all_words = []
    for tweet in tweets:
        for word in tweet:
            all_words.append(word)

    return all_words


def build_dataset(words, vocabulary_size, unk_label=0):
    # Build dictionary of word -> order of frequency

    count = [["UNK", -1]]
    count.extend(Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = unk_label  # dictionary["UNK"]
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def tweet_to_int(tweet, dictionary, unk_label=0) -> list:
    # Map a tweet to integers based on supplied dictionary

    tweet_int = []
    unk_count = 0

    for word in tweet:
        if word in dictionary:
            idx = dictionary[word]
        else:
            idx = unk_label
            unk_count += 1
        tweet_int.append(idx)

    return tweet_int


# Preprocess data
# Load data
twitter = pd.read_csv("/home/slazien/PycharmProjects/FuseMind/TwitteRNN/data/Sentiment_Analysis_Dataset.csv",
                      error_bad_lines=False)

tweets = twitter["SentimentText"]
sent = twitter["Sentiment"]

# Clean text
cleaned_tweets = clean_tweets(tweets)

# Tokenize tweets
tokenized_tweets = tokenize_tweets(cleaned_tweets, n_cores=8)

# Transform tweets into a list of words
words = get_words(tokenized_tweets)

# Finally, prepare data for the model
# vocab_size = int(0.01*len(words))  # take 10% of all words in the tweets
vocab_size = 100000

# Set unknown label to -1 instead of 0 since keras uses 0 to pad sequences
data, count, dictionary, reverse_dictionary = build_dataset(words, vocab_size, unk_label=0)
del data

# Transform tweets to format accepted by LSTM
X = []
X_lengths = []
for tweet in tokenized_tweets:
    mapping = tweet_to_int(tweet, dictionary, unk_label=0)
    X.append(mapping)
    X_lengths.append(len(mapping))
X = np.asarray(X)
y = np.asarray(sent)

# Split data intro training and validation
seed = 42
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=seed)

# Truncate and pad input sequences
max_seq_len = max(X_lengths)
X_train = sequence.pad_sequences(X_train, maxlen=max_seq_len)
X_val = sequence.pad_sequences(X_val, maxlen=max_seq_len)

# Define and train the model, yay! :D
embed_vec_len = 16
train_n = len(X_train)
model = Sequential()
model.add(Embedding(vocab_size, embed_vec_len, input_length=max_seq_len))
model.add(Convolution1D(filters=64, kernel_size=3, padding="same", activation="relu"))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())
history = model.fit(X_train[:], y_train[:], epochs=1, batch_size=64, validation_split=0.2)
plt.plot(history.history["val_acc"])
