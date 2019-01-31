import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import copy
from tqdm import tqdm_notebook

def get_splits(bdf,y_col,cv=5,val_size=0.1,test_size=0.2):
    df=copy.copy(bdf)
 
    def my_train_test_split(cdf,y_col,test_size=test_size):
        adf=copy.copy(cdf)
        train_df,val_df,_,_=train_test_split(adf,adf[y_col],test_size=test_size,random_state=42,shuffle=True)
        train_ix=list(train_df.index.values)  
        val_ix=list(val_df.index.values)
        return train_ix,val_ix
        
    if cv>0:
        skf = StratifiedKFold(n_splits=cv,random_state=42,shuffle=True)
        skf.get_n_splits(df, df[y_col])
        splits=[a for a in skf.split(df,df[y_col])]
        
        
        if val_size==0:
            return splits
        
        else:
            all_splits=[]
            for train_ix,test_ix in splits:
                train_df=df.loc[train_ix,:]
                
                new_train_ix,val_ix=my_train_test_split(train_df,y_col=y_col,test_size=val_size)
                all_splits.append((new_train_ix,val_ix,test_ix))
            return all_splits
    else:
        train_ix,test_ix=my_train_test_split(df,y_col=y_col,test_size=test_size)
        if val_size==0:
            return[(train_ix,test_ix)]
        else:
            new_train_ix, val_ix=my_train_test_split(df,test_size=val_size)
            return[(new_train_ix,val_ix,test_ix)]
        

def get_split_dicts(splits):    
    dsplits=[]
    for split in splits:
        splitd={}
        for ixs in split:
            splitd['train']={'ixs':split[0]}
            splitd['test']={'ixs':split[-1]}
            if len(split)==3:
                splitd['val']={'ixs':split[1]}
        dsplits.append(splitd)
    return dsplits

def get_split_dfs(dsplits,df):
    for dsplit in dsplits:
        dsplit['train']['df']=df.iloc[dsplit['train']['ixs'],:]
        dsplit['test']['df']=df.iloc[dsplit['test']['ixs'],:]
        try:
            dsplit['val']['df']=df.iloc[dsplit['val']['ixs'],:]
        except KeyError:
            pass
    return dsplits
        


def encode_tweet(tweet,encoder):
    seq=[encoder[word] for word in tweet.split()]
    seq=[ix for ix in seq if ix!=-1 ]
    return seq


#TODOO add remove duplicatesz
def stack_df(bdf,x_cols,y_col,new_col_name='stacked'):
    df=copy.copy(bdf)
    cdf=pd.melt(df, id_vars=y_col, value_vars=x_cols, value_name=new_col_name)
    return cdf

def get_arrays(bdf,encoder,x_cols,y_col,max_len,stack=True,channels=False):
    df=copy.copy(bdf)
    if stack==True:
        stackdf=stack_df(df,x_cols=x_cols,y_col=y_col)
         
        xarr=[]
        #turn each word into a sequence of indexes
        for tweet in stackdf['stacked']:
            xarr.append(encode_tweet(tweet,encoder))
        xarr=np.array(xarr)
        #pad sequences
        xarr=pad_sequences(xarr,maxlen=max_len)
        #gte y array
        yarr=np_utils.to_categorical(stackdf[y_col])
        
    if channels==True:
        langs=[]
        for col in x_cols:
            xarr=[]
            for tweet in df[col]:
                xarr.append(encode_tweet(tweet,encoder))
            xarr=pad_sequences(xarr,maxlen=max_len)
            langs.append(xarr)
        xarr=langs
        yarr=np_utils.to_categorical(df[y_col])
    return xarr,yarr
        
def get_fold_arrays1(folds,encoder,x_cols,y_col,max_len,channels=False):
    for fold in folds:
        for split,_ in fold.items():
            xarr,yarr=get_arrays(fold[split]['df'],encoder=encoder,channels=channels,x_cols=x_cols,y_col=y_col,max_len=max_len)
            fold[split]['xarr']=xarr
            fold[split]['yarr']=yarr
    return folds

def get_fold_arrays(folds,encoder,x_cols,y_col,max_len,test_orginal=False,train_channels=False,test_channels=False):
    
    for fold in folds:
        #train
        xarr,yarr=get_arrays(fold['train']['df'],encoder=encoder,channels=train_channels,x_cols=x_cols,y_col=y_col,max_len=max_len)
        fold['train']['xarr']=xarr
        fold['train']['yarr']=yarr

        #val
        try:
            xarr,yarr=get_arrays(fold['val']['df'],encoder=encoder,channels=train_channels,x_cols=x_cols,y_col=y_col,max_len=max_len)
            fold['val']['xarr']=xarr
            fold['val']['yarr']=yarr
        except KeyError:
            pass
        #test
        xarr,yarr=get_arrays(fold['test']['df'],encoder=encoder,channels=test_channels,x_cols=x_cols,y_col=y_col,max_len=max_len)
        fold['test']['xarr']=xarr
        fold['test']['yarr']=yarr
        #if test_original==True:
           #fold['test']['xarr']=xarr[0]         
    return folds

def prep_data(df,x_cols,y_col,encoder,max_len,cv=5,val_size=0.1,test_size=0.2,test_original=False,train_channels=False,
              test_channels=False):
    splits=get_splits(df,y_col=y_col,cv=cv,val_size=val_size,test_size=test_size)
    folds=get_split_dicts(splits)
    folds=get_split_dfs(folds,df=df)
    folds=get_fold_arrays(folds,encoder,x_cols=x_cols,y_col=y_col,max_len=max_len,train_channels=train_channels, test_channels=test_channels)#test_original=test_original
    return folds

def embedding_array(encoder,vec_dict, oov_zeros=True,oov_random=False):
    embed_dim=len(list(vec_dict.values())[0])
    #create corresponding embedding matrix
    embed_len=len([ix for ix in list(encoder.values()) if ix!=-1])
    if oov_zeros==True:
        embedding_matrix = np.zeros((embed_len, embed_dim))
    if oov_random==True:
        embedding_matrix=random_embedding_array(embed_len,embed_dim)
    print('creating embedding array...')
    for word, ix in tqdm_notebook(encoder.items()):
        if ix>0:
            #embedding_vector = embeddings_index.get(word)
            try:
                embedding_matrix[ix] = vec_dict[word]
            except KeyError:
                pass
            
    return embedding_matrix

def random_embedding_array(embed_len,dim=300):
    return np.random.uniform(low=-1,high=1,size=(embed_len,dim))