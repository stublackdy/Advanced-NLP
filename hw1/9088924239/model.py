import torch
import torch.nn as nn
import zipfile
import numpy as np
from tqdm import tqdm

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path, weights_only=False)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    # print(emb_file)
    with open(emb_file, "r") as f:
        embeddings=f.read()

    
    emb_list=embeddings.strip().split("\n")
    # print(emb_list[0].split(" "))

    emb_dict={}

    for i in emb_list:
        emb_dict[i.split()[0]]=np.array(i.split()[1:], dtype=np.float32)

    final_emb=[]

    # print(len(vocab))
    # print(vars(vocab))
    # print(dir(vocab))
    for i in range(len(vocab)):

        word=vocab.id2word[i]
        # print(word)
        
        if word in emb_dict:
            final_emb.append(emb_dict[word]) 

        else:
            final_emb.append(np.zeros(emb_size, dtype=np.float32))        ## Taking 0 for the unknown words in the vocab
            # final_emb.append(np.random.uniform(-0.25, 0.25, emb_size))

    
    return np.array(final_emb)


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

        else:
            self.embedding_layer=nn.Embedding(len(self.vocab), self.args.emb_size)

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        

        self.model = nn.Sequential(nn.Linear(self.args.emb_size, 256),
                            nn.ReLU(),
                            nn.Linear(256, 128),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(128, 64),
                            nn.ReLU(),
                            nn.Dropout(p=0.2),
                            nn.Linear(64, self.tag_size))

    def init_model_parameters(self, v=0.08):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        for i in self.model.parameters():
            nn.init.uniform_(i, -v, v)
        

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """

        np_embeddings=load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        tensor_emb = torch.from_numpy(np_embeddings)

        self.embedding_layer = nn.Embedding(len(self.vocab), self.args.emb_size)

        with torch.no_grad():
            self.embedding_layer.weight.data.copy_(tensor_emb)

        # self.embedding_layer=nn.Embedding.from_pretrained(tensor_emb, padding_idx=0, freeze=False)


    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """

        mask_list=[]

        for i in x:
            mask=[]
            for j in range(len(i)):
                if i[j]==0:
                    mask.append(0)
                    
                else:
                    mask.append(1)

            mask_list.append(mask)
                 

        # print(x.shape)
        # print(x)
        # print(mask_list)
        mask_list_tensor=torch.tensor(mask_list)

        embeddings=self.embedding_layer(x)

        # print(embeddings.shape)
        # print(mask_list_tensor.shape)
        expanded_mask_list_tensor=mask_list_tensor.unsqueeze(-1).expand(-1, -1, self.args.emb_size)
        # print(mask_list_tensor)
        # print(mask_list_tensor.shape)

        # print((embeddings*expanded_mask_list_tensor)[0])
        # print((embeddings*expanded_mask_list_tensor).shape)

        # print(expanded_mask_list_tensor.sum(dim=1).shape)

        pooled_embedding=(embeddings*expanded_mask_list_tensor).sum(dim=1)/expanded_mask_list_tensor.sum(dim=1)

        score=self.model(pooled_embedding)

        

        return score
