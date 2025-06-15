"""
Author: @jay.shah
Date: 6/15/2025
"""
import re
import numpy as nppip
from data import text

#

"""
The first two functions here are part of the preprocessing step for making my text data usable.
I am going to tokenize the text and then map those token to indices (and vice versa).
"""

def tokenize(text: str) -> list:
    """
    I can't feed raw string text into the word2vec model I'm building.
    Therefore I'll have to pre-process the text through tokenization. I'm learning that's the way to go when working with NLP models. I can use regex to achieve this.
    """
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def map_tokens_and_indices(tokens: list) -> dict:
    """
    Another useful operation is creating a map between tokens and indices.
    Here I'll create a function to map all tokens and their indices and vice versa.
    This is essentially a lookup table, which will be useful for when I do one-hot encoding.
    """

    word_to_id = {}
    id_to_word = {}

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word

# I'll call my tokenize function on the example text data
tokens = tokenize(text)

# Next I'll call my word to id and id to work mapping function to see how my text is being mapped
word_to_id, id_to_word = map_tokens_and_indices(tokens)

# I can print the output of the mapping function using
# print(word_to_id) and print(id_to_word)

"""
Now I can start the step of actually generating my training data.
My training data is going to take the form of matrices. My tokens are still in the form of strings,
so I need to use a technique called one-hot vectorizatino to encode them numerically.

I also need to generate a bundle of target and input values because word2vec is a supervised learning technique.

Word2vec is some kind of blackbox magic, but rather a result of careful training with input and output values, just like any other machine learning task.

So here comes the crux of word2vec, I'm going to loop through each word (or token) in a sentence. In each sentence I am going to look at words to the left and right of the input token/word and map the relationship.

For the example text: "little dark back-alleys behind the", I would use a sliding window function to identify the following pairs around "back-alleys":
    ["back-alleys", "little"]
    ["back-alleys", "dark"]
    ["back-alleys", "behind"]
    ["back-alleys", "the"]

Note: The window size is 2, which is why i'm looking up two words to the left and right behind the input text. In this way, I'm forcing the model to get a rough understanding of the context of a word, i.e. which words stick together. In my example data, I'm guessing a lot of reoccurring names will be assoicated together, I'm hoping the model I'm making should capture this relationship.
"""

def generate_training_data(tokens: list, word_to_id: dict, window: int) -> list:
    """
    Generating two lists, x and y, for the center word and context words respectively.
    The separation of these two lists will aid in matrice operations down the road.
    """
    X = []
    y = []
    n_tokens = len(tokens)

    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i),
            range(i, min(n_tokens, i + window + 1))
        )

        for j in idx:
            if i == j:
                continue
            # X encodes the center word
            X.append(one_hot_encode(word_to_id[tokens[i]], len(word_to_id)))
            # y encodes context words
            y.append(one_hot_encode(word_to_id[tokens[j]], len(word_to_id)))
    
    return np.asarray(X), np.asarray(y)

def concat(*iterables):
    """
    Yields a concatenation of a range of indexes based on the sliding window for each token.
    """
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id, vocab_size):
    """
    Generates an array for each token, where the token's index is indicated with a 1
    """
    res = [0] * vocab_size
    res[id] = 1
    return res

# Now I can generate some training data with a window size of 2
X, y =  generate_training_data(tokens, word_to_id, window=2) 

# Let's check the dimensionality of the data to get an idea of the matrices I'll be working with:
# print(X.shape)
# print(y.shape
# Printed:
# (18606, 1573)
# (18606, 1573)
# 1573 is the number of unique tokens, 18606 is the number of traning examples, this number increases as the window for context increases

"""
Now, let's implement the code to train the model.
"""

