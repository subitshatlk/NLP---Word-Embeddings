# Mini-project 1: Training a CBoW Model

## Project Overview
This repository contains the implementation and resources for training a Continuous Bag of Words (CBoW) model as introduced by Mikolov et al. (2013). The aim of this mini-project is to train a CBoW model to generate embeddings for a predetermined vocabulary. You will find all necessary data, scripts, and instructions for running the model, training it, and evaluating the embeddings.

## CBoW Model Description
The CBoW model predicts a target word based on the context words surrounding it. It includes a projection layer that maps one-hot encoded vectors to a low-dimensional embedding space, and a classifier layer that predicts the target word from the embedding.

## Configurations
- Learning Rates: 0, 0.01, 0.001, 0.0001
- Embedding Dimension: 100
- Batch Size: Recommended 64
- Maximum Epochs: 10
- Context Size: 5

## Project Overview
In this mini-project, the primary goal was to train and evaluate a Continuous Bag of Words (CBoW) model, a method first introduced by Mikolov et al. in 2013. This type of neural network model is designed to predict a target word based on the context of surrounding words without the word order in the input. The process involves projecting context words into a low-dimensional space and then using these embeddings to predict the hidden (target) word.

The implementation of the CBoW model consisted of several steps. Initially, a vocabulary was defined, and a mapping was created to associate each word with a unique index. This setup facilitated the transformation of textual data into numerical form that the model could process. Data was then loaded and preprocessed, involving tokenization and conversion into one-hot encoded vectors, which represent each word as a distinct dimension in a high-dimensional space.

A significant part of the project was focused on the model architecture, which includes a projection layer and a classifier layer. The projection layer converts the one-hot encoded vector of context words into dense embeddings using a learned projection matrix. These embeddings are then summed to represent the context for a target word. The classifier layer subsequently attempts to predict the target word from these summed embeddings.

For training, the CBoW model utilized Cross Entropy Loss to optimize the prediction of the target word, testing various learning rates to find the most effective one for minimizing the loss on a development dataset. The best-performing model was saved and used to compute embeddings for each word in the vocabulary, which were then evaluated on specific tasks such as word similarity and analogy resolution. These tasks tested the quality of the embeddings by measuring how well they captured semantic and syntactic relationships between words.

Additionally, an exploratory part of the project involved analyzing the embeddings using visualization techniques such as PCA or SVD to project the high-dimensional word vectors into a two-dimensional space. This exercise aimed to provide intuitive insights into how words are related in the embedding space, although it was also noted that such a reduction can sometimes be misleading due to the loss of information.
