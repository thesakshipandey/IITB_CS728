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
import copy, time

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
        # we gotta extract NER tags first...
        # ner_tag_names = self.data_train.features['ner_tags'].feature.names
        # self.tag_map = {name:index for name, index in enumerate(ner_tag_names)}
        # self.inverse_tag_map = {index:name for name, index in enumerate(ner_tag_names)} 


    def add_embeddings(self, ds, word_embedding_mapping, embedded_feature, embedding_name, embedding_dim):
        def f(ex):
            ex[embedding_name] = [
                word_embedding_mapping.get(tok, np.zeros(embedding_dim, dtype=np.float32)) for tok in ex[embedded_feature]
            ]
            return ex
        return ds.map(f, desc='adding embedding')

    def load_embedding_into_dataset(self, input_json_wkm, input_nparray_kem, embedding_name='glove', embedding_dim = 200):
        # Quick note: The Glove embeddings were created using variable sized context windows 
        # The SVD embeddings were generated using term frequency counts across docs 
        
        word_embedding_mapping = collections.defaultdict()
        if embedding_name == 'glove':
            word_key_mapping = collections.defaultdict()
            key_embedding_mapping = collections.defaultdict()
            with open(input_json_wkm, 'r') as f:
                word_key_mapping = json.load(f)
            key_embedding_mapping = np.load(input_nparray_kem, allow_pickle=True).item()
            for item in word_key_mapping.keys():
                k = word_key_mapping[item] 
                word_embedding_mapping[item] = key_embedding_mapping[k]

        elif embedding_name == 'svd':
            with open(input_json_wkm, 'r') as f:
                word_embedding_mapping = json.load(f)

        # print(len(word_embedding_mapping['0']))
        # time.sleep(10)
        
    
        self.data_train = self.add_embeddings(self.data_train, word_embedding_mapping, 'tokens', 'emb', embedding_dim=embedding_dim)
        self.data_test = self.add_embeddings(self.data_test, word_embedding_mapping, 'tokens', 'emb', embedding_dim)
        self.data_val = self.add_embeddings(self.data_val, word_embedding_mapping, 'tokens', 'emb', embedding_dim) 

        # self.data_train = self.add_embeddings(self.data_train, self.tag_map, 'ner_tags', 'label', embedding_dim=1)
        # self.data_test = self.add_embeddings(self.data_test, self.tag_map, 'ner_tags', 'label', embedding_dim=1)
        # self.data_val = self.add_embeddings(self.data_train, self.tag_map, 'ner_tags', 'label', embedding_dim=1)

        return


    

        




class FeedforwardNN(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        layers = []
        prev = params['input_dim']
        activation = torch.nn.ReLU() if params['activation'] == 'relu' else torch.nn.Tanh()
        for h in params['hidden_dim']:
            layers += [
                torch.nn.Linear(prev, h),
                activation,
                torch.nn.Dropout(params['dropout'] if 'dropout' in params else 0.1)
            ]
            prev = h
        layers.append(torch.nn.Linear(prev, params['output_dim']))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)
    def predict(self, x):
        return torch.softmax(self.forward(x), dim=-1)

PAD_LABEL = -100

def collateFunction(batch):
    batch_size = len(batch)
    max_sentence_length = max(len(ex['emb']) for ex in batch)
    d = len(batch[0]['emb'][0])

    X = torch.zeros(batch_size, max_sentence_length, d, dtype=torch.float32)
    y = torch.full((batch_size, max_sentence_length), PAD_LABEL, dtype=torch.long)

    for i, ex in enumerate(batch):
        assert all(len(v)==len(ex["emb"][0]) for v in ex["emb"])
        s = len(ex['emb'])
        X[i, :s] = torch.tensor(np.asarray(ex['emb']), dtype=torch.float32)
        y[i, :s] = torch.tensor(ex['ner_tags'], dtype=torch.long) 
    return X.view(-1, d), y.view(-1) # this flattens the 3D tensor to become a 2D tensor and does so along the 3rd axis


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
            dataHandler.load_embedding_into_dataset('token_ids.json', f'context_window_{context_window}_dimension_{embedding_dim}_final_word_embeddings.npy', embedding_name='glove', embedding_dim=embedding_dim)
            num_classes = dataHandler.data_train.features['ner_tags'].feature.num_classes
            default_ffnn_params['output_dim'] = num_classes
            default_ffnn_params['input_dim'] = embedding_dim
            train_loader = DataLoader(dataHandler.data_train, batch_size=512, shuffle=True, collate_fn=collateFunction)
            test_loader = DataLoader(dataHandler.data_test, batch_size=512, shuffle=False, collate_fn=collateFunction)
            model = FeedforwardNN(params=default_ffnn_params)
            id2label = dataHandler.data_train.features["ner_tags"].feature.names
            model.id2label = {i: name for i, name in enumerate(id2label)}
            losses = train(model, train_loader, test_loader, epochs=25)
            plot_loss(losses, title=f'loss v epochs MLP with GLOVE{embedding_dim}x{context_window}', save_name=f'losses_glove_{embedding_dim}x{context_window}.png')
            results_glove[f'{context_window} + {embedding_dim}'] = eval_sklearn(model, test_loader, id2label=id2label)
    with open("glove_results.json", "w") as f:
        json.dump(results_glove, f)
    
    results_svd = {}

    for ipdim in [50, 100, 200, 300]:
        dataHandler = ThisClassDoesThingsToData()
        dataHandler.load_embedding_into_dataset(f'term_embeddings_svd_non-clubbed_{ipdim}.json', 'random.json', embedding_name='svd', embedding_dim=ipdim)
        num_classes = dataHandler.data_train.features['ner_tags'].feature.num_classes
        train_loader = DataLoader(dataHandler.data_train, batch_size=512, shuffle=True, collate_fn=collateFunction)
        test_loader = DataLoader(dataHandler.data_test, batch_size=512, shuffle=False, collate_fn=collateFunction)
        default_ffnn_params['output_dim'] = num_classes
        default_ffnn_params['input_dim'] = ipdim
        model = FeedforwardNN(params=default_ffnn_params)    
        id2label = dataHandler.data_train.features["ner_tags"].feature.names
        model.id2label = {i: name for i, name in enumerate(id2label)} 
        losses = train(model, train_loader, test_loader, epochs=30)
        plot_loss(losses, title='loss v epochs MLP dropout 0.1', save_name=f'losses_svd_{ipdim}.png')
        results_svd[ipdim] = eval_sklearn(model, test_loader, id2label=id2label)
    with open("svd_results.json", "w") as f:
        json.dump(results_svd, f)