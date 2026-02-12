import torch 
from torch.nn.functional import mse_loss, kl_div, softmax
from datasets import load_dataset
import nltk
from tqdm import tqdm
import os

from sklearn.metrics import precision_recall_fscore_support, classification_report

from torch.utils.data import DataLoader, Dataset

import collections, json
import matplotlib.pyplot as plt
import copy, time, sys


from functools import partial

import numpy as np


default_ffnn_params = {
    'input_dim': 200,
    'output_dim': 9,
    'hidden_dim': [1024],
    'activation': 'relu',
    'dropout': 0.01
}






class ThisClassDoesThingsToData:

    def __init__(self):
        data = load_dataset(
        "eriktks/conll2003",
            trust_remote_code=True,
            token=os.getenv("HUGGING_FACE_TOKEN"),  # only if your environment needs auth
        )
        self.data_val = data['validation']
        self.data_train = data['train']
        self.data_test = data['test']
        self.embedding = None
        # we gotta extract NER tags first...
        # ner_tag_names = self.data_train.features['ner_tags'].feature.names
        # self.tag_map = {name:index for name, index in enumerate(ner_tag_names)}
        # self.inverse_tag_map = {index:name for name, index in enumerate(ner_tag_names)} 


    def add_embeddings(self, ds, word_key_mapping, unk_id):
        def f(ex):
            ex['tokens'] = [
                word_key_mapping.get(tok, unk_id) for tok in ex['tokens']
            ]
            return ex
        return ds.map(f, desc='adding embedding')

    def load_embedding_into_dataset(self, input_json_wkm, input_nparray_kem, embedding_name='glove', embedding_dim = 200):
        # Quick note: The Glove embeddings were created using variable sized context windows 
        # The SVD embeddings were generated using term frequency counts across docs 
        
        word_embedding_mapping = collections.defaultdict()
        if embedding_name != 'glove':
            sys.exit(0)
        word_key_mapping = collections.defaultdict()
        key_embedding_mapping = collections.defaultdict()
        with open(input_json_wkm, 'r') as f:
            word_key_mapping = json.load(f)
        key_embedding_mapping = np.load(input_nparray_kem, allow_pickle=True).item()
        for item in word_key_mapping.keys():
            k = word_key_mapping[item] 
            word_embedding_mapping[item] = key_embedding_mapping[k]
        
        PAD_ID_EMB = len(key_embedding_mapping)
        UNK_ID = PAD_ID_EMB + 1

        embedding_weights = np.random.normal(0, 0.02, size = (len(key_embedding_mapping)+2, embedding_dim)).astype(np.float32)
        if PAD_ID_EMB is not None:
            embedding_weights[PAD_ID_EMB] = 0.0

        for k, v in key_embedding_mapping.items():
            i = int(k)
            vec = np.asarray(v, dtype=np.float32)
            if vec.shape != (embedding_dim, ):
                raise ValueError(f'id {i} dimension mismatch') # yes i debug like this. no this is not a gpt failsafe
            if 0 <= i < PAD_ID_EMB:
                embedding_weights[i] = vec
        embedding_weights[UNK_ID] = embedding_weights.mean(axis=0)
        
        # print(len(word_embedding_mapping['0']))
        # time.sleep(10)


        
    
        self.data_train = self.add_embeddings(self.data_train, word_key_mapping, UNK_ID)
        self.data_test = self.add_embeddings(self.data_test, word_key_mapping, UNK_ID)
        self.data_val = self.add_embeddings(self.data_val, word_key_mapping, UNK_ID) 

        # self.data_train = self.add_embeddings(self.data_train, self.tag_map, 'ner_tags', 'label', embedding_dim=1)
        # self.data_test = self.add_embeddings(self.data_test, self.tag_map, 'ner_tags', 'label', embedding_dim=1)
        # self.data_val = self.add_embeddings(self.data_train, self.tag_map, 'ner_tags', 'label', embedding_dim=1)

        return torch.tensor(embedding_weights, dtype=torch.float32), PAD_ID_EMB


    

        




class NNWithTrainableEmbeddings(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        layers = []
        prev = params['input_dim']
        activation = torch.nn.ReLU() if params['activation'] == 'relu' else torch.nn.Tanh()

        self.embedding = torch.nn.Embedding(params['vocab_size'] + 2, params['input_dim'], padding_idx=params['PAD_ID_EMB'])

        with torch.no_grad():
            self.embedding.weight.copy_(params['embedding_weights'])
        
        self.embedding.weight.requires_grad = True

        for h in params['hidden_dim']:
            layers += [
                torch.nn.Linear(prev, h),
                activation,
                torch.nn.Dropout(params['dropout'] if 'dropout' in params else 0.1)
            ]
            prev = h
        layers.append(torch.nn.Linear(prev, params['output_dim']))
        self.net = torch.nn.Sequential(*layers)
    
    def forward(self, token_ids):
        token_ids = token_ids.long()                 
        x = self.embedding(token_ids)                
        x = x.view(-1, x.size(-1))                   
        logits = self.net(x)                         
        return logits
    
    def predict(self, x):
        return torch.softmax(self.forward(x), dim=-1)

PAD_LABEL = -100

def collateFunction(batch, pad_id_emb, pad_label=-100):
    B = len(batch)
    Lmax = max(len(ex["tokens"]) for ex in batch)

    X = torch.full((B, Lmax), pad_id_emb, dtype=torch.long)
    y = torch.full((B, Lmax), pad_label, dtype=torch.long)

    for i, ex in enumerate(batch):
        s = len(ex["tokens"])
        X[i, :s] = torch.tensor(ex["tokens"], dtype=torch.long)
        y[i, :s] = torch.tensor(ex["ner_tags"], dtype=torch.long)

    return X.view(-1), y.view(-1) 

############### Interesting note
# in pytorch, nn.functional.cross_entropy automatically softmaxes the logits and then computes CE loss in a numerically
# stable way. This is recommended for training. 


def eval_sklearn(model, loader, id2label=None):
    device = next(model.parameters()).device
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)

            pred = logits.argmax(dim=-1)
            mask = (y != PAD_LABEL)

            y_true.append(y[mask].detach().cpu())
            y_pred.append(pred[mask].detach().cpu())

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    print(f"macro P/R/F1: {p_macro:.4f} / {r_macro:.4f} / {f1_macro:.4f}")
    print(f"micro P/R/F1: {p_micro:.4f} / {r_micro:.4f} / {f1_micro:.4f}")

    if id2label is not None:
        labels = list(range(len(id2label)))
        target_names = [id2label[i] for i in labels]
        print(classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0
        ))

    return p_macro, r_macro, f1_macro, classification_report(y_true, y_pred, labels=labels, target_names=target_names, zero_division=0)

def train(model, train_loader,  test_loader=None, epochs=10, lr=1e-3, wd=1e-2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    losses = []

    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=PAD_LABEL)

            optimizer.zero_grad(set_to_none=False)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mask = (y != PAD_LABEL)
                total_loss += loss.item()*mask.sum().item()
                total_correct += ((logits.argmax(-1) == y) & mask).sum().item()
                total_n += mask.sum().item()

            
        print(f'loss at epoch {epoch} is {total_loss/total_n} with accuracy {total_correct/total_n}')
        losses.append(total_loss/total_n)

    if test_loader is not None:
        model.eval()
        v_loss, v_correct, v_n = 0.0, 0, 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(logits, y, ignore_index=PAD_LABEL)
                mask = (y != PAD_LABEL)
                v_loss += loss.item()*mask.sum().item()
                v_correct += ((logits.argmax(dim=-1) == y) & mask).sum().item()
                v_n += mask.sum().item()
            print(f'val loss {v_loss/v_n} val acc {v_correct/v_n}')
            id2label = model.id2label if hasattr(model, "id2label") else None
            eval_sklearn(model, test_loader, id2label=id2label)
    return losses




        
def cross_validation_and_grid_search():
    pass

def plot_loss(losses, title, save_name):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('objective')
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()
    return














######################################## DATA PROCESSING PIPELINE ##########################################








if __name__ == '__main__':
    ########################## EXPERIMENTS ###################################################
    # 1. Glove, with different dims 

    results_glove = {}

    for context_window in [5, 10, 15]:
        for embedding_dim in [50, 100, 200,300]:
            if(context_window == 15 and embedding_dim == 300):
                continue
            dataHandler = ThisClassDoesThingsToData()
            embedding_weights, vocab_size = dataHandler.load_embedding_into_dataset('token_ids.json', f'context_window_{context_window}_dimension_{embedding_dim}_final_word_embeddings.npy', embedding_name='glove', embedding_dim=embedding_dim)
            num_classes = dataHandler.data_train.features['ner_tags'].feature.num_classes
            default_ffnn_params['output_dim'] = num_classes
            default_ffnn_params['input_dim'] = embedding_dim
            default_ffnn_params['embedding_weights'] = embedding_weights
            default_ffnn_params['PAD_ID_EMB'] = vocab_size
            default_ffnn_params['vocab_size'] = vocab_size
            train_loader = DataLoader(dataHandler.data_train, batch_size=512, shuffle=True, collate_fn=partial(collateFunction, pad_id_emb=vocab_size, pad_label=-100))
            test_loader = DataLoader(dataHandler.data_test, batch_size=512, shuffle=False, collate_fn=partial(collateFunction, pad_id_emb=vocab_size, pad_label=-100))
            model = NNWithTrainableEmbeddings(params=default_ffnn_params)
            id2label = dataHandler.data_train.features["ner_tags"].feature.names
            model.id2label = {i: name for i, name in enumerate(id2label)}
            losses = train(model, train_loader, test_loader, epochs=25)
            plot_loss(losses, title=f'loss v epochs EMB+MLP with GLOVE{embedding_dim}x{context_window}', save_name=f'losses_glove_EMB+MLP_{embedding_dim}x{context_window}.png')
            results_glove[f'{context_window} + {embedding_dim}'] = eval_sklearn(model, test_loader, id2label=id2label)
    with open("trained_glove_results.json", "w") as f:
        json.dump(results_glove, f)
    
