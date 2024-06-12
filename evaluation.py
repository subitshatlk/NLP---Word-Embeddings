import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import spacy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

def read_embeddings(embeddings_file):
    word_vectors = {}
    with open(embeddings_file, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip().split()
        num_words, embedding_dim = int(first_line[0]), int(first_line[1])
        for line in file:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=float)
            word_vectors[word] = vector
    return num_words, embedding_dim, word_vectors


def compute_analogy(word_vectors, word1, word2, word3):
    if word1 not in word_vectors or word2 not in word_vectors or word3 not in word_vectors:
        return "One or more words not found in embeddings"
    
    # Calculate the analogy: word1 - word2 + word3
    result_vector = word_vectors[word2] - word_vectors[word1] + word_vectors[word3]
    
    # Find the most similar word to the result using cosine similarity
    most_similar_word = None
    max_similarity = -1.0
    
    for word in word_vectors:
        if word not in [word1, word2, word3]:
            similarity = cosine_similarity([result_vector], [word_vectors[word]])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_word = word
    
    return most_similar_word

def compare_word_pairs(word_vectors, word_pairs):
    similarities = []
    for pair in word_pairs:
        pair_similarity = []
        for word1, word2 in pair:
            if word1 in word_vectors and word2 in word_vectors:
                vector1 = word_vectors[word1]
                vector2 = word_vectors[word2]
                similarity = cosine_similarity([vector1], [vector2])[0][0]
                pair_similarity.append(similarity)
            else:
                pair_similarity.append(None)  # Handle cases where words are not found in embeddings
        similarities.append(pair_similarity)
    return similarities


    
def printing_similarities(word_pairs,similarities):
    for i, pair in enumerate(word_pairs):
        print(f"Pair {i + 1}:")
        for (word1, word2), similarity in zip(pair, similarities[i]):
            if similarity is not None:
                print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
            else:
                print(f"One or both of the words '{word1}' and '{word2}' not found in embeddings.")

def computing_analogies(analogies):
    for analogy in analogies:
        word1, word2, word3 = analogy
        result_word = compute_analogy(word_vectors, word1, word2, word3)
        print(f"{word1}:{word2}, {word3}:{result_word}")


def plot_two_dim():
    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # List of words
    words = ["horse", "cat", "dog", "I", "he", "she", "it", "her", "his", "our", "we", "in", "on",
             "from", "to", "at", "by", "man", "woman", "boy", "girl", "king", "queen", "prince", "princess"]

    # Get word embeddings for the words in the list
    word_list = []
    for word in words:
        word_vector = nlp(word).vector
        word_list.append(word_vector)

    word_list = np.array(word_list)

    # Perform dimensionality reduction using PCA
    pca = PCA(n_components=2)
    word_embeddings_2d = pca.fit_transform(word_list)

    # Plot the 2-D projection
    plt.figure(figsize=(10, 8))
    plt.scatter(word_embeddings_2d[:, 0], word_embeddings_2d[:, 1], marker='o', s=50)

    for i, word in enumerate(words):
        plt.annotate(word, xy=(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]), fontsize=12)

    plt.title("2-D Projection of Word Embeddings using PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()

    # Save the plot as an image (e.g., PNG)
    plt.savefig("/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp1/mp1_release/models/word_embeddings_pca_plot.png")

    # Show the plot (optional)
    plt.show()




if __name__ == "__main__":
    embeddings_file = '/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp1/mp1_release/models/word_embeddings13.txt'  # Replace with the path to your embeddings file
    num_words, embedding_dim, word_vectors = read_embeddings(embeddings_file)
    print(f"Number of words: {num_words}, Embedding dimension: {embedding_dim}")

    word_pairs = [
        (['cat', 'tiger'], ['plane', 'human']),
        (['my', 'mine'], ['happy', 'human']),
        (['happy', 'cat'], ['king', 'princess']),
        (['ball', 'racket'], ['good', 'ugly']),
        (['cat', 'racket'], ['good', 'bad'])
    ]

    similarities = compare_word_pairs(word_vectors, word_pairs)
    printing_similarities(word_pairs,similarities)

    word_pairs_test = [
        (['avoid', 'omit'], ['child', 'kids']),
        (['beautiful', 'beauty'], ['disliked', 'hatred']),
        (['politics', 'government'], ['before', 'after'])]

    similarities_test = compare_word_pairs(word_vectors,word_pairs_test)
    printing_similarities(word_pairs_test,similarities_test)
    
    # for i, pair in enumerate(word_pairs):
    #     print(f"Pair {i + 1}:")
    #     for (word1, word2), similarity in zip(pair, similarities[i]):
    #         if similarity is not None:
    #             print(f"Similarity between '{word1}' and '{word2}': {similarity:.4f}")
    #         else:
    #             print(f"One or both of the words '{word1}' and '{word2}' not found in embeddings.")

    analogies = [
    ("king", "queen", "man"),
    ("king", "queen", "prince"),
    ("king", "man", "queen"),
    ("woman", "man", "princess"),
    ("prince", "princess", "man")
]

    computing_analogies(analogies)

    analogies_test = [
    ("man", "woman", "grandfather"),
    ("king", "queen", "groom"),
    ("woman", "man", "nurse")
]

    computing_analogies(analogies_test)
    plot_two_dim()

    


    


    
    


