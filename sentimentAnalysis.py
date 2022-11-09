import random
from textcleaner import *
import nltk
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pdb

normalizer = WordNetLemmatizer()
stop_words = stopwords.words('english')

def sentimentValue(text, llist):
    goodEmote = []
    mehEmote = []
    badEmote = []

    with open('GMB/good.txt', 'r')as fg:
        data = fg.readlines()
        for line in data:
            newline = line.strip('\n')
            goodEmote.append(newline)

        for words in goodEmote:
            word_tokenize(words)

    with open('GMB/meh.txt', 'r')as fm:
        data = fm.readlines()
        for line in data:
            newline = line.strip('\n')
            mehEmote.append(newline)
        for words in mehEmote:
            word_tokenize(words)

    with open('GMB/bad.txt', 'r')as fb:
        data = fb.readlines()
        for line in data:
            newline = line.strip('\n')
            badEmote.append(newline)
        for words in badEmote:
            word_tokenize(words)

    # Uncomment All Comments Under This One to Run Standalone
    sentiment = 0
    tokenized = word_tokenize(text)
    cleaned = cleaner(text)
    data = list(cleaned.split(' '))
    #print(tokenized)
    normalized = ' '.join([normalizer.lemmatize(wordList) for wordList in tokenized])
    # print(normalized)
    partOfSpeech(normalized)
    statement = [word for word in tokenized if word not in stop_words]
    for words in statement:
        randnum = random.randrange(1)
        if words in goodEmote:
            sentiment += 1

        elif words in mehEmote:
            if randnum == 1:
                sentiment += 1

            else:
                pass

        elif words in badEmote:
            sentiment -= 1

        else:
            pass

    # print(statement)  # This One
    average = sentiment / len(llist)
    # print(round(average))  # This Two
    return round(average, 3)

def partOfSpeech(reply):
        pos = []
        for words in reply:
            probable_part_of_speech = wordnet.synsets(words)
            pos_counts = Counter()
            pos_counts["n"] = len(
                [item for item in probable_part_of_speech if item.pos() == "n"])
            pos_counts["v"] = len(
                [item for item in probable_part_of_speech if item.pos() == "v"])
            pos_counts["a"] = len(
                [item for item in probable_part_of_speech if item.pos() == "a"])
            pos_counts["r"] = len(
                [item for item in probable_part_of_speech if item.pos() == "r"])

            most_likely_part_of_speech = pos_counts.most_common(1)[0][0]
            pos.append(most_likely_part_of_speech)
        return pos


# humanreply = str(input("Human:  ")).lower()   # This Three
# sentimentValue(humanreply)    # This Four
