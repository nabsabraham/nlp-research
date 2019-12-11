# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 18:53:36 2019

@author: Nabila Abraham
"""

import torch
import numpy as np
from pytorch_transformers import *

model_class = BertModel
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'

# Load pretrained model/tokenizer
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

# Encode text
def get_embeddings(sentence):
    input_ids = torch.tensor([tokenizer.encode(sentence, 
                                               add_special_tokens=True)])
    last_hidden_states = model(input_ids)[0][0]
    return last_hidden_states.detach().numpy()

# Similarity metric 
def cosine_sim(x,y):
    dotprod = np.dot(x,y)
    normx = np.linalg.norm(x)
    normy = np.linalg.norm(y)
    return (dotprod) / (normx * normy)

def compute_paraphrase_similarity(s1,s2):
    embed1 = get_embeddings(s1)[0]
    #embed1 = embed1.sum(axis=0)
    
    embed2 = get_embeddings(s2)[0]
    #embed2 = embed2.sum(axis=0)
    
    similarity = cosine_sim(embed1, embed2)
    return similarity
    
    
s1 = "Charlie Chan is off the case for the Fox Movie Channel."
s2 = "The Fox Movie Channel has banned Charlie Chan."
s3 = """Assessment of current business conditions improved substantially, 
        the Conference Board said, jumping to 55 from 40 in the first quarter."""
s4 = """Feelings about current business conditions improved substantially 
        from the first quarter, jumping from 40 to 55."""

sim = compute_paraphrase_similarity(s1,s3)