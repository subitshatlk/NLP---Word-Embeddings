import argparse
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
import os


def get_eval_stats(emb_file):   
    wv = KeyedVectors.load_word2vec_format(datapath(emb_file), binary=False)
    sim = wv.evaluate_word_pairs("/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp1/mp1_release/test_files/wordsim_similarity_goldstandard.txt")
    analogy = wv.evaluate_word_analogies("/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp1/mp1_release/test_files/questions-words_headered.txt")
    print(f"Word Similarity Test Pearson Correlation: {sim[0][0]}")
    print(f"Accuracy on Analogy Test: {analogy[0]}")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_file", default="/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp1/mp1_release/models/word_embeddings13.txt", type=str)
    args = vars(parser.parse_args())
    get_eval_stats(args["emb_file"])

