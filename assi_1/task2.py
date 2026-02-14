# %%
import numpy as np
import json
import re
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import svds

# %%
data = json.load(open("vocab_dict.json", "r"))

# %%
word_list = []
rows = []   # word indices
cols = []   # document indices
vals = []   # counts
doc_lenghts = []


top_k = 100
task2 = True  # if true then task 2 otherwise task 5

word2ind = {}

docId2ind = {}

words_encountered = {}

clubbed = False  # Don't change for assignment task 2 -> Assignment terms changed

for word in data.keys():
    passages = data[word]
    if clubbed:
        word = word.lower()
    if word in words_encountered:
        scanned_passages_for_given_word = words_encountered[word]
    else:
        scanned_passages_for_given_word = set()
        word_list.append(word)
        word2ind[word] = len(word_list) - 1
    for passage in passages:
        if passage[0] in scanned_passages_for_given_word:
            print(f"Already encountered this passage for this word: {word}")
            continue
        rows.append(word2ind[word])
        if passage[0] in docId2ind:
            colId = docId2ind[passage[0]]
        else:
            colId = len(docId2ind)
            docId2ind[passage[0]] = colId
            doc_lenghts.append(len(passage[1].split()))
        cols.append(colId)
        scanned_passages_for_given_word.add(passage[0])
        if clubbed:
            vals.append(int(len(re.findall(rf"\b{re.escape(word)}\b", passage[1], re.IGNORECASE))))
        else:
            vals.append(int(len(re.findall(rf"\b{re.escape(word)}\b", passage[1]))))
    words_encountered[word] = scanned_passages_for_given_word

assert(max(cols) == len(doc_lenghts) -1)

m = len(word_list)
n = max(cols) + 1
assert(rows[-1] == len(word_list) - 1)

term_doc = coo_matrix((vals, (rows, cols)), shape=(m, n))

# %%
if task2:
    tf_idf = term_doc
else:
    tf = term_doc/np.array(doc_lenghts)
    eidf = len(doc_lenghts)/term_doc.getnnz(axis=1)
    idf = np.log(eidf)
    tf_idf = tf.multiply(idf.reshape(-1,1))

# %%
U_k, S_k, Vt_k = svds(tf_idf, k=top_k)
idx = np.argsort(S_k)[::-1] #top-k are returned in reverse order
U_k, S_k, Vt_k = U_k[:, idx], S_k[idx], Vt_k[idx]

term_proj = U_k@np.diag(S_k)  #taking projection in lower-space
term_proj /= np.linalg.norm(term_proj, axis=1, keepdims=True)  #normalizing for cosine similarity

word_embeddings = {
    word: term_proj[i].tolist()
    for i, word in enumerate(word_list)
}

with open(f"term_embeddings_svd_{'task2' if task2 else 'task5'}_{top_k}.json", "w") as f:
    json.dump(word_embeddings, f)

# %%
# nn = NearestNeighbors(
#     n_neighbors=5,
#     metric="cosine",
#     algorithm="brute"
# )

# nn.fit(term_proj)

# %%
# test_words = ["India", "INDIA", "pakistan", "PAKISTAN", "Nepal", "China", "CHINA" ,"MINNEAPOLIS", "Minneapolis"]
# word_list = np.array(word_list)

# %%
# distances, indices = nn.kneighbors([term_proj[word2ind[w]] for w in test_words])

# for i, w in enumerate(test_words):
#     print(f"Five words closest to {w}: {word_list[indices[i]]}")

# %%
# distances, indices = nn.kneighbors([term_proj[word2ind[w]] for w in test_words])

# for i, w in enumerate(test_words):
#     print(f"Five words closest to {w}: {word_list[indices[i]]}")


