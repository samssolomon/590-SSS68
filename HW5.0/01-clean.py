
import os
import numpy as np
import nltk
from nltk.corpus import wordnet
import string
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#------------------------------------------------------#
### Load: 
## "Frankenstein" by Mary Wollstonecraft Shelley (Book 1)
## "Pride and Prejudice" by Jane Austen (Book 2)
## "The Great Gatsby" by F Scott Fitzgerald (Book 3)
#------------------------------------------------------#

book1 = open("Frankenstein.txt").read().lower()
book1 = nltk.tokenize.sent_tokenize(book1)

book2 = open("Pride and Prejudice.txt").read().lower()
book2 = nltk.tokenize.sent_tokenize(book2)

book3 = open("The Great Gatsby.txt").read().lower()
book3 = nltk.tokenize.sent_tokenize(book3)

# Combine words of all books into one corpus which will be tokenized

corpus = book1 + book2 + book3

# Generate labels

book1_labels = np.repeat(1,len(book1))
book2_labels = np.repeat(2,len(book2))
book3_labels = np.repeat(3,len(book3))

corpus_labels = np.concatenate([book1_labels, book2_labels, book3_labels])



#------------------------------------------------------#
# Tokenize vocabulary of all books
#------------------------------------------------------#

maxlen = 30
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(corpus)
sequences_all = tokenizer.texts_to_sequences(corpus)
word_index = tokenizer.word_index
print('Found %s unique tokens in entire corpus.' % len(word_index))

corpus_tokenized = pad_sequences(sequences_all, maxlen=maxlen, padding = 'post')


#------------------------------------------------------#
# Write tokenized corpus and its labels to txt files
#------------------------------------------------------#

np.savetxt('corpus_tokenized.txt',corpus_tokenized, fmt="%10.5f")
print("Saved tokenized corpus.")

np.savetxt('corpus_labels.txt',corpus_labels, fmt='%s')
print("Saved tokenized labels.")
