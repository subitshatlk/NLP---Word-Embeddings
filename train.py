import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


from scripts.utils import get_files, get_word2ix, process_data 


class CBOW_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW_Model, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
        )

    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis=1)
        x = self.linear(x)
        return x

def preprocess_data(dataset):
    data = process_data(dataset, args.context_size, word2idx)
    data_list = [item for sublist in data for item in sublist]

    context_window = args.context_size
    batch_input = []
    batch_output = []

    for i in range(context_window, len(data_list) - context_window):
        target_word = data_list[i]
        context = []

        for j in range(i - context_window, i + context_window + 1):
            if j != i and j >= 0 and j < len(data_list):
                context.append(data_list[j])

        batch_input.append(context)
        batch_output.append(target_word)

    return TensorDataset(torch.tensor(batch_input), torch.tensor(batch_output))

def load_data(in_dir):
    train_files = get_files(f'{in_dir}/data/train')
    train_dataset = preprocess_data(train_files)
    dev_files = get_files(f'{in_dir}/data/dev')
    dev_dataset = preprocess_data(dev_files)

    return train_dataset, dev_dataset


def print_dataset(dataset):
    for inputs, labels in dataset:
        print("Inputs:", inputs)
        print("Labels:", labels)


def save_embeddings(embeddings,word2idx,file_path):
    word_to_embedding = {word: embedding for word, embedding in zip(word2idx, embeddings)}
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"{embeddings.shape[0]} {embeddings.shape[1]}\n")
        for word, embedding in word_to_embedding.items():
            embedding_str = ' '.join(map(str, embedding))
            file.write(f"{word} {embedding_str}\n")

    



def train_process(train_data, dev_data, vocab_size, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_epochs = int(args.epochs)

    model = CBOW_Model(vocab_size, int(args.embeddings_dim)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))

    best_perf_dict = {"metric": 0, "epoch": 0}
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=args.batch_size)

    for ep in range(1, max_epochs + 1):
        print(f"Epoch {ep}")

        # Training loop
        train_loss = []
        

        for inp, lab in tqdm(train_loader):
            model.train()
            optimizer.zero_grad()
            out = model(inp.to(device))
            loss = loss_fn(out, lab.to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.cpu().item())

        print(f"Average training batch loss: {np.mean(train_loss)}")

        # Evaluation
        dev_loss_list = []
        gold_labels = []
        predicted_labels = []

        
        with torch.no_grad():
            for inp, lab in tqdm(dev_loader):
                model.eval()
                out = model(inp.to(device))
                preds = torch.argmax(out, dim=1)
                predicted_labels.extend(preds.cpu().tolist())
                gold_labels.extend(lab.tolist())

                dev_loss = loss_fn(out, lab.to(device))
                dev_loss_list.append(dev_loss.cpu().item())

        dev_f1 = f1_score(gold_labels, predicted_labels, average='macro')
        print(f"Dev Loss: {np.mean(dev_loss_list)}")
        print(f"Dev F1: {dev_f1}\n")

        # Update the `best_perf_dict` if the best dev performance seen
        # so far is beaten
        if dev_f1 > best_perf_dict["metric"]:
            best_perf_dict["metric"] = dev_f1
            best_perf_dict["epoch"] = ep
            checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'dev_metric': dev_f1,
        'epoch': ep
    } 

    # Save the model checkpoint to a file
    torch.save({"model_param": model.state_dict(),
                "optim_param": optimizer.state_dict(),
                "dev_loss": dev_loss,
                "dev_metric": dev_f1,
                "epoch": ep}, '/scratch/general/vast/u1471339/cs6957/assignment1/models/model')

    print(f"\nBest Dev performance of {best_perf_dict['metric']} at epoch {best_perf_dict['epoch']}")


    embeddings = list(model.parameters())[1]
    embeddings = embeddings.cpu().detach().numpy()
    return embeddings

    # Save the trained model or its parameters here if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="/models")
    parser.add_argument('--in_dir', type=str, default="/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp1/mp1_release")
    parser.add_argument('--vocab_dir', type=str, default="vocab.txt")
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embeddings_dim', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--context_size', type=int, default=5)

    args, _ = parser.parse_known_args()

    # Define constants and hyperparameters
    word2idx = get_word2ix(args.vocab_dir)
    vocabulary_size = len(word2idx)

    
    training_data, development_data = load_data(args.in_dir) 

    train_model = train_process(training_data, development_data, vocabulary_size, args)

    output_file_path = '/uufs/chpc.utah.edu/common/home/u1471339/u1471339_mp1/mp1_release/models/word_embeddings13.txt' 
    save_embeddings(train_model,word2idx,output_file_path)



    
