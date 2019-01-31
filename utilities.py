import pandas as pd
import copy
from keras import backend as K

#found below on stack exchange!
def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def load_df(data_filep,x_cols,y_col,y_map=None):
    
    def get_label_ix(x):
        return y_map[x]

    df=pd.read_csv(data_filep)
    if y_map!=None:
        df[y_col[0]]=df[y_col[0]].apply(get_label_ix)
    
    if 'exists' in list(df):
        df=df[df['exists']==True]
        df=df.drop(labels=['exists'],axis=1)
    cols=copy.copy(x_cols)
    cols.extend(y_col)
    if cols:
        df=df[cols]
    return df.reset_index(drop=True)

def word_has_vec(word,vecs):
    try:
        vecs[word]
        return True
    except KeyError:
        return False

def get_embedding_encoder(vocab,vec_dict):
    vocab_ix={'<pad>':0}
    ix_vocab={0:'<pad>'}
    i=1
    for word in vocab:
        if word_has_vec(word,vec_dict):
            vocab_ix[word]=i
            ix_vocab[i]=word
            i+=1
        else:
            vocab_ix[word]=-1
            ix_vocab[-1]=word
    return vocab_ix, ix_vocab

def get_vocab_encoder(vocab):
    vocab_ix={'<pad>':0}
    ix_vocab={0:'<pad>'}
    i=1
    for word in vocab:
        vocab_ix[word]=i
        ix_vocab[i]=word
        i+=1
    return vocab_ix,ix_vocab

def load_embedding_file(embeddings_fp):
    def get_embeddings(word, *vec): 
        return word, np.asarray(vec, dtype='float32')
    embeddings_index = {}
    print('loading glove vecs (approx 5 mins) ...')
    for line in tqdm_notebook(open(embeddings_fp),total=2196017):
        word,vec = get_embeddings(*line.rstrip().rsplit(' '))
        embeddings_index[word]=vec
    return embeddings_index