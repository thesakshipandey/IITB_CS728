import time
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from utils import read_json, damping_function, make_token_ids, get_token_ids
from tqdm import tqdm

# function to fill the co-occurence matrix...
def filing_co_occurence_matrix(w: int, data: dict, co_occurence_matrix: dict, words_in_vocab: set[str]):
    for key in data.keys():
        for temp in data[key]:
            passage = temp[1].split()
            for index, word in enumerate(passage):
                if word == key and word in words_in_vocab:
                  if word == "e":
                     print("yes")
                  i = 0
                  j = 0
                  while i != w and j != w:
                    if index - i >= 0 and passage[index-i] in words_in_vocab:
                      co_occurence_matrix[word, passage[index-i]] += 1
                    if index + j < len(passage) and passage[index+j] in words_in_vocab:
                      co_occurence_matrix[word, passage[index+j]] += 1
                    i += 1
                    j += 1

# loading the data...
data = read_json("vocab_dict.json")
doc_ids = set()
words_in_vocab = set()

for key in data.keys():
  for temp in data[key]:
    doc_ids.add(temp[0])
  words_in_vocab.add(key)

print(f'Number of documents in the corpus are {len(doc_ids)}')
print(f'Number of words in the corpus are {len(words_in_vocab)}')

n = len(words_in_vocab)

# defining the hyperparameters...
w_s = [5, 10, 15] # context window...
dims = [50, 100, 200, 300] # embedding dimension...
x_max = 200
alpha = 0.75
learning_rate = 0.05 # took from the paper and works well for dimension less than 300...
eps = 1e-8 # standard...
# epochs = [10, 20, 30, 40, 50] # number of epochs for training...

# creating the token ids for the words in the vocab...
make_token_ids(words_in_vocab)
token_ids = get_token_ids()

for w in w_s:
    # creating the co-occurence matrix...
    co_occurence_matrix = defaultdict(int)
    filing_co_occurence_matrix(w, data, co_occurence_matrix, words_in_vocab)
    # checking the length of the co-occurence matrix...
    print(f"The length of the co-occurence matrix is {len(co_occurence_matrix)}")
    for d in dims:
        # defining the trainable parameters...
        U = np.random.randn(n, d).astype(np.float32)
        V = np.random.randn(n, d).astype(np.float32)

        b = np.zeros(n, dtype=np.float32)
        b_ = np.zeros(n, dtype=np.float32)

        final_word_embeddings = {}

        epochs = 0

        grad_square_sum_U = np.zeros(U.shape, dtype=np.float32)
        grad_square_sum_V = np.zeros(V.shape, dtype=np.float32)
        grad_square_sum_b = np.zeros(b.shape, dtype=np.float32)
        grad_square_sum_b_ = np.zeros(b_.shape, dtype=np.float32)

        loss_per_epoch = []
        latency_per_epoch = []


        total_epochs = 50
        pbar = tqdm(total=total_epochs, desc='Processing')
        while epochs < total_epochs:
            t0 = time.time() # for the latency calculation...
            epoch_loss = 0.0 # for monitoring the epoch loss...
            for (word1, word2) in co_occurence_matrix.keys():
                if co_occurence_matrix[(word1, word2)] > 0:
                    damping_function_value = damping_function(co_occurence_matrix[(word1, word2)], x_max)

                    common_grad_term = 2 * damping_function_value * (np.dot(U[token_ids[word1], :], V[token_ids[word2], :]) + b[token_ids[word1]] + b_[token_ids[word2]] - np.log(co_occurence_matrix[(word1, word2)]))
                    grad_wrt_u = common_grad_term * V[token_ids[word2], :]
                    grad_wrt_v = common_grad_term * U[token_ids[word1], :]
                    grad_wrt_b = common_grad_term
                    grad_wrt_b_ = common_grad_term

                    epoch_loss += damping_function_value * (np.dot(U[token_ids[word1], :], V[token_ids[word2], :]) + b[token_ids[word1]] + b_[token_ids[word2]] - np.log(co_occurence_matrix[(word1, word2)])) ** 2

                    # accumulating the squared gradients...
                    grad_square_sum_U[token_ids[word1], :] += grad_wrt_u ** 2
                    grad_square_sum_V[token_ids[word2], :] += grad_wrt_v ** 2
                    grad_square_sum_b[token_ids[word1]] += grad_wrt_b ** 2
                    grad_square_sum_b_[token_ids[word2]] += grad_wrt_b_ ** 2

                    # updating the parameters...
                    U[token_ids[word1], :] -= (learning_rate/(eps + np.sqrt(grad_square_sum_U[token_ids[word1], :])) * grad_wrt_u)
                    V[token_ids[word2], :] -= (learning_rate/(eps + np.sqrt(grad_square_sum_V[token_ids[word2], :])) * grad_wrt_v)

                    b[token_ids[word1]] -= (learning_rate/(eps + np.sqrt(grad_square_sum_b[token_ids[word1]])) * grad_wrt_b)
                    b_[token_ids[word2]] -= (learning_rate/(eps + np.sqrt(grad_square_sum_b_[token_ids[word2]])) * grad_wrt_b_)
            epochs += 1
            pbar.update(1)
            loss_per_epoch.append(epoch_loss)
            latency_per_epoch.append(time.time() - t0)
            print(f"Epoch number {epochs} completed, Loss = {epoch_loss:.4f}, Latency = {latency_per_epoch[-1]:.2f}s")

        plt.figure(figsize=(8,5))
        plt.plot(range(1, epochs+1), loss_per_epoch, marker='o')
        plt.xlabel("Epoch")
        plt.ylabel("Total Loss")
        plt.title(f"GloVe Training Loss per Epoch for context window = {w} and dimension = {d}")
        plt.grid(True)
        plt.savefig(f"glove_training_loss_context_window_{w}_dimension_{d}.png", dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(8,5))
        plt.plot(range(1, epochs+1), latency_per_epoch, marker='o', color='orange')
        plt.xlabel("Epoch")
        plt.ylabel("Time (s)")
        plt.title(f"Epoch Latency for context window = {w} and dimension = {d}")
        plt.grid(True)
        plt.savefig(f"epoch_latency_context_window_{w}_dimension_{d}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        for key in token_ids.keys():
            # if performance hurts then tell me...
            final_word_embeddings[token_ids[key]] = U[token_ids[key], :] + V[token_ids[key], :]

        # storing this in a pickle file...
        np.save(f"./context_window_{w}_dimension_{d}_final_word_embeddings.npy", final_word_embeddings, allow_pickle=True)