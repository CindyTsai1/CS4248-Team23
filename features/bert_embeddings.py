import pandas as pd

def bert_embeddings_feature(prefix):
    embeddings = pd.read_csv("features/" + prefix + "_bert_embeddings.csv")[['0']] 
    embeddings['0'] = embeddings['0'].apply(lambda x: x[3:-2]) # take away [[ ....]] the brackets
    embeddings = embeddings['0'].str.split(expand=True).astype(float) # split string, convert type 
    
    return embeddings