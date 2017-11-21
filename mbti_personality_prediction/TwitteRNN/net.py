import re
import pickle
from collections import Counter
from multiprocessing import Pool

import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, LSTM, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class TwitteRNN:

    def __init__(self, vocab_size, unk_label=0, n_cores=8, max_seq_len="max"):
        self.data = None
        self.tweets = None
        self.cleaned_tweets = None
        self.tokenized_tweets = None
        self.words = None
        self.X = None
        self.y = None
        self.max_seq_len = None
        self.n_examples = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.unk_label = unk_label
        self.dictionary = None
        self.reverse_dictionary = None
        self.unk_count = None

        self.model = None

        self.n_cores = n_cores

        self.functions = [
            self.clean_tweets,
            self.tokenize_tweets,
            self.get_words,
            self.build_mapping,
            self.map_all
        ]

    def clean_tweets(self, tweets, sent, vocab_size):
        # Take a list of tweets and output cleaned tweets list
        cleaned_tweets = []

        self.tweets = tweets
        self.y = np.asarray(sent)
        self.vocab_size = vocab_size

        for tweet in self.tweets:
            tweet_temp = " ".join(tweet.split())  # strip duplicate whitespace
            tweet_temp = re.sub("\.{4,}", "...", tweet_temp)  # replace multiple ..... with ...
            tweet_temp = re.sub(r"^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?", "",
                                tweet_temp)  # strip URLs

            # LaTeX
            tweet_temp = re.sub("\$\$[^$$]+\$\$", '', tweet_temp)  # $$...$$
            tweet_temp = re.sub("\$[^$]+\$", '', tweet_temp)  # $...$
            tweet_temp = re.sub(r'\\begin\{(.*?)\}(.*?)\\end\{\1\}', '', tweet_temp)  # \begin...\end
            tweet_temp = re.sub('\[([^]]+)\]', '', tweet_temp)  # [...]

            tweet_temp = re.sub(r"[@#$%^&*~]", "", tweet_temp)  # remove some weird characters
            tweet_temp = tweet_temp.lower()  # to lowercase
            """ replace 3+ consecutive occurrences of a char by "n" chars to preserve 'excitement';
            watch out with that! since it might change the vocabulary for words such as 'too' etc."""
            matches = re.findall(r"((\w)\2{2,})", tweet_temp)
            n = 2
            if matches is not None:
                for match in matches:
                    tweet_temp = re.sub(match[0], match[1] * n, tweet_temp)

            # newlines
            tweet_temp = tweet_temp.replace("\\n", "").replace("\\r", "").replace("\n", "").\
                replace("\r", "").replace("\\", "")

            cleaned_tweets.append(tweet_temp)

        self.cleaned_tweets = cleaned_tweets

        return self.cleaned_tweets

    def tokenize_tweets(self, chunk_size=10000):
        # Tokenize tweets using multiple CPU cores if needed
        tokenized_tweets = []

        bar = tqdm(total=len(self.cleaned_tweets))
        pool = Pool(processes=self.n_cores, maxtasksperchild=1)
        for text in pool.imap(word_tokenize, self.cleaned_tweets, chunksize=chunk_size):
            tokenized_tweets.append(text)
            bar.update()

        self.tokenized_tweets = tokenized_tweets

        return self.tokenized_tweets

    def get_words(self):
        # Concatenate all tweets into one huge list of words (needed for word freq counts)
        all_words = []
        for tweet in self.tokenized_tweets:
            for word in tweet:
                all_words.append(word)

        self.words = all_words

        return self.words

    def build_mapping(self):
        # Build dictionary of (word -> order of frequency) mapping
        count = [["UNK", -1]]
        count.extend(Counter(self.words).most_common(self.vocab_size - 1))
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        self.dictionary = dictionary
        dict_name = str(self.vocab_size) + "_dict.pickle"
        with open(dict_name, "wb") as f:
            pickle.dump(self.dictionary, f)
        self.reverse_dictionary = reverse_dictionary

        return self.dictionary

    def map_all(self):
        # Transform tweets to format accepted by LSTM
        X = []
        X_lengths = []
        for tweet in self.tokenized_tweets:

            mapping = []
            for word in tweet:
                if word in self.dictionary:
                    idx = self.dictionary[word]
                else:
                    idx = self.unk_label

                mapping.append(idx)
            X.append(mapping)
            X_lengths.append(len(mapping))

        X = np.asarray(X)
        self.X = X

        if self.max_seq_len == "max":
            self.max_seq_len = max(X_lengths)

        self.n_examples = X.shape[0]

        return self.X

    def preprocess(self, tweets, sent):
        self.data = self.clean_tweets(tweets, sent, self.vocab_size)
        for function in self.functions[1:len(self.functions)]:
            print("Running: " + str(function))
            self.data = function()

    def create_model(self, embedded_vec_len=16, filters=32, kernel_size=3, padding='same', pool_size=2, dropout=0.5):
        # Define and train the model, yay! :D
        model = Sequential()
        model.add(Embedding(self.vocab_size, embedded_vec_len, input_length=self.max_seq_len))
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, activation='relu'))
        model.add(MaxPooling1D(pool_size=pool_size))
        model.add(LSTM(100))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation='sigmoid'))

        print(model.summary())

        self.model = model

    def train(self, test_size=0.2, seed=42, train_n=30000, epochs=3, batch_size=256,
              early_stop_delta=0.01, early_stop_patience=3):

        try:
            # Split data intro training and validation
            self.X_train, self.X_val, self.y_train, self.y_val = \
                train_test_split(self.X, self.y, test_size=test_size, random_state=seed)
        except TypeError:
            print("Please run the 'preprocess' method first")

        # Truncate and pad input sequences
        self.X_train = sequence.pad_sequences(self.X_train, maxlen=self.max_seq_len)
        self.X_val = sequence.pad_sequences(self.X_val, maxlen=self.max_seq_len)

        # Define callbacks
        filepath = "weights-improvement-{epoch:02d}-{val_acc:.3f}.hdf5"
        checkpointer = ModelCheckpoint(filepath=filepath,
                                       monitor="val_acc", verbose=1, save_best_only=True, mode="max")
        early_stopper = EarlyStopping(monitor="val_acc", min_delta=early_stop_delta,
                                      patience=early_stop_patience, verbose=1)
        callbacks_list = [checkpointer, early_stopper]

        try:
            model = self.model
        except TypeError:
            print("Please create the model first using the 'create_model' method")

        model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(self.X_train[:train_n], self.y_train[:train_n], epochs=epochs, batch_size=batch_size,
                  validation_split=0.2, callbacks=callbacks_list)

        self.model = model

    def load_model(self, weights_path, dict_filename):
        # Load weights from supplied hdf5 file and word dictionary for prediction
        try:
            self.model.load_weights(weights_path)
        except AttributeError:
            print("Please create the model first using the 'create_model' method")

        try:
            with open(dict_filename, "rb") as f:
                self.dictionary = pickle.load(f)
        except IOError:
            print("Word dictionary with given name not found")

    def predict(self, tweet, print_tokenized=False):
        # Preprocess the tweet
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
                tweet_temp = re.sub(match[0], match[1] * n, tweet_temp)

        cleaned_tweet = tweet_temp
        tokenized_tweet = word_tokenize(cleaned_tweet)

        if len(tweet_temp.split()) == 0:
            return -1

        mapping = []
        unk_count = 0

        if print_tokenized:
            print(tokenized_tweet)

        # Map tweet
        for word in tokenized_tweet:
            if word in self.dictionary:
                idx = self.dictionary[word]
            else:
                idx = self.unk_label
                unk_count += 1
            mapping.append(idx)
            self.unk_count = unk_count

        mapping = np.asarray(mapping)

        # Truncate and pad input sequence
        padded_mapping = sequence.pad_sequences([mapping], maxlen=self.max_seq_len)

        pred = self.model.predict(padded_mapping)

        return pred[0][0]

    def evaluate(self, X_test, y_test):
        model_eval = self.model.evaluate(X_test, y_test)
        print("Accuracy: %.2f%%" % (model_eval[1]*100))
