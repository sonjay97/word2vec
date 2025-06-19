"""
Author: @jay.shah
Date: 6/15/2025
"""
import re
import numpy as np
import matplotlib.pyplot as plt
from data import text





np.random.seed(42)

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

def concat(*iterables: list) -> list[int]:
    """
    Yields a concatenation of a range of indexes based on the sliding window for each token.
    """
    for iterable in iterables:
        yield from iterable

def one_hot_encode(id: int, vocab_size: int) -> list:
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

def init_network(vocab_size: int, n_embedding: int) -> dict:
    """
    The values of the dictionary I'm creating are the weight matrices and keys.
    """
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }

    return model

# Initialize the model with 10 dimensions for each word
model = init_network(len(word_to_id), 10)

# Now I can code the function for forward propagation
def forward(model: dict, X: list, return_cache=True) -> dict:
    """
    Forward propagation code, I'm performing the three layers of matrix multiplication 
    to achieve the embedded matrix, weighted matrix, and softmax matrix.
    I will use cache for backward propagation as an intermediate variable that holds calculation answers
    """

    cache = {}
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])

    if not return_cache:
        return cache["z"]
    return cache

def softmax(X):
    res = []
    for x in X:
        exp = np.exp(x)
        res.append(exp / exp.sum())
    return res

# checking the shape of the models to verify the dimnesions and token lenght
# print((X @ model["w1"]).shape)
# print((X @ model["w1"] @ model["w2"]).shape)

def backward(model, X, y, alpha):
    """
    Backward propagation training. I'll do my best to explain each step as I go.
    """
    # start with getting the forward values, using the results of forward propagation I can use matrix multiplication to backwards propagate the intermediate matrices and update their values according to how they should be adjusted
    cache = forward(model, X, True)
    # Calculate the error between the predicted ouput and the actual output
    # cache z is the model's prediction
    # y is one hot encoded correct values
    # both have the same shape (batch_size, vocab_size)
    da2 = cache["z"] - y 
    # Calculates the error gradient for w2, which is the weight layer between the embedding layer and the output layer
    # cache a1 is the activation from the first layer after the input, not necessarily an embedding layer, all the one hot vectors for each token
    # .T transposes, essnetially pivoting the rows and columns
    # we transpose a1 because we need the same shapes of the matrices to perform multiplication
    # So basically we're taking what the model thinks to do, and how wrong it is, and put that together. Now we know how much to shift the weights of the embed layer a1
    dw2 = cache["a1"].T @ da2
    # dw1 propagates error back through the second layer
    da1 = da2 @ model["w2"].T
    # dw1 computes how the first weight matrix should change based on the original input x and the backpropagated error
    dw1 = X.T @ da1

    assert(dw2.shape == model["w2"].shape)
    assert(dw1.shape == model["w1"].shape)

    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2

    return cross_entropy(cache["z"], y)

def cross_entropy(z, y):
    """z is the softmax output, y is the one hot encoded TRUE label, or our accurate one hot matrix"""
    return - np.sum(np.log(z) * y)


n_iter = 50
learning_rate = 0.05

history = [backward(model, X, y, learning_rate) for _ in range(n_iter)]

plt.plot(range(len(history)), history, color="skyblue")
# plt.show()

def get_embedding(model, word):
    """function that grabds the embeddings for a given word"""
    try:
        idx = word_to_id[word]
    except KeyError:
        print("`word` not in corpus")
    one_hot = one_hot_encode(idx, len(word_to_id))

    return forward(model, one_hot)["a1"]

# example call for the word "machine"
print(get_embedding(model, "machine"))