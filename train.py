import os
import pandas as pd
import numpy as np
import datetime
import time
import re

from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from keras import backend as K
from keras.models import load_model
from statistics import mode
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint


class Train():
    
    def __init__(self,model,model_name,summary,x_cols,modeldir='./models/'):
        self.x_cols=x_cols
        self.model=model
        self.summary=summary
        self.modeldir=modeldir
        self.model_name=model_name
        
        
    


    def train(self,data,model,path,total_epochs=50,batch_size=1000,monitor_score='val_f1',monitor_mode='auto',patience=5):#TOODO Change this so that data_seqs are in the constructor
        self.monitor_score=monitor_score
        self.monitor_mode=monitor_mode
        self.total_epochs=total_epochs
        self.batch_size=batch_size
        self.patience=patience
        self.path=path
        
        
        val=False
        X_train=data['train']['xarr']
        y_train=data['train']['yarr']
        try:
            X_val=data['val']['xarr']
            y_val=data['val']['yarr']
            val=True
        except KeyError:
            pass

        filepath=self.path+"_e{epoch:02d}_"+monitor_score+"-{"+monitor_score+":.2f}.hdf5"
        #filepath=self.path+"_e{epoch:02d}_acc-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(os.path.join(self.modeldir,filepath), 
                                     monitor=monitor_score,
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode=monitor_mode)
        earlystop = EarlyStopping(monitor=monitor_score, min_delta=0.00005, patience=patience, mode=monitor_mode)

        starttime=time.time()
        if val==True:
            model.fit(X_train,y_train,
                      validation_data=(X_val,y_val),
                      batch_size=batch_size, 
                      epochs=total_epochs,
                      verbose=1,
                      callbacks=[earlystop,checkpoint])
        else:
            model.fit(X_train,y_train,
                  batch_size=batch_size, 
                  epochs=total_epochs,
                  verbose=1,
                  callbacks=[earlystop,checkpoint])
        endtime=time.time()
        train_time=endtime-starttime
        self.train_time=train_time/60
        


    def get_best_model(self):
        model_filenames=[fn for fn in os.listdir(self.modeldir) if fn.startswith(self.path)]
        model_filenames.sort()
        if len(model_filenames)==1:
            best_fn=model_filenames[0]
        else:
            remove_filenames=model_filenames[:-1]
            best_fn=model_filenames[-1]
            for fn in remove_filenames:
                os.remove(os.path.join(self.modeldir,fn))
        self.best_fn=best_fn



      #TODO finish writing metrics etc to dataframe
    def evaluate(self,data,y_map,tracker_fn='tracker.csv',channel_probs=False,predict_english_only=False):
        print(channel_probs)
        best_epochs=int(re.findall('_e(\d+)_',self.best_fn)[0])
   
        try:
            model = load_model(os.path.join(self.modeldir,self.best_fn))
        except:
            model = load_model(os.path.join(self.modeldir,self.best_fn),custom_objects={'f1': ut.f1})

        y_decoder={v: k for k, v in y_map.items()}
        X_test=data['test']['xarr']
        y_test=data['test']['yarr']
        y_test_labels=[np.argmax(cat) for cat in y_test]
        #predict
        #y_pred = model.predict(data.X_test_seq,batch_size=self.batch_size)
        
        
        if channel_probs==True:
            all_lang_preds=[]
            for arr in X_test:
                y_pred = model.predict(arr,batch_size=self.batch_size)
                #y_pred_labels=[np.argmax(pred) for pred in y_pred]
                all_lang_preds.append(y_pred)
                #all_lang_preds.append(y_pred_labels)

            y_pred_labels=[]
            for i,probs in enumerate(zip(*all_lang_preds)):
                y_pred_labels.append(np.argmax(np.average(np.array(probs),axis=0)))
                
                
        elif predict_english_only==True:
            y_pred = model.predict(X_test[0],batch_size=self.batch_size)
            y_pred_labels=[np.argmax(pred) for pred in y_pred]
          
        else:
            y_pred = model.predict(X_test,batch_size=self.batch_size)
            y_pred_labels=[np.argmax(pred) for pred in y_pred]

        #evaluate
        conf_matrix=confusion_matrix(y_test_labels, y_pred_labels)
        acc=accuracy_score(y_test_labels, y_pred_labels)
        w_prec,w_rec,w_f1,_=precision_recall_fscore_support(y_test_labels, y_pred_labels,
                                                            average='weighted')
        mic_prec,mic_rec,mic_f1,_=precision_recall_fscore_support(y_test_labels, y_pred_labels,
                                                                  average='micro')
        mac_prec,mac_rec,mac_f1,_=precision_recall_fscore_support(y_test_labels, y_pred_labels,
                                                                  average='macro')
        results={'model_filepath':self.best_fn,
                 'languages':self.x_cols,
                 'model_name':self.model_name,
                 'date':'-',
                 'time':'-',
                 'train_time':self.train_time,
                 'best_epochs':best_epochs,
                 'accuracy':acc,
                 'batch_size':self.batch_size,
                 'total_epochs': self.total_epochs,
                 'summary':self.summary,
                 'precision (weighted)':w_prec, 
                 'f1 (weighted)':w_f1,
                 'precision (micro)':mic_prec,
                 'recall (micro)':mic_rec,
                 'f1 (micro)':mic_f1,
                 'precision (macro)':mac_prec,
                 'recall (macro)':mac_rec,
                 'f1 (macro)':mac_f1,
                 'confusion_matrix':conf_matrix}

        tracker=pd.read_csv(os.path.join(self.modeldir,tracker_fn))
        tracker=tracker.append(results, ignore_index=True)
        tracker.to_csv(os.path.join(self.modeldir,tracker_fn),index=False)
    #TODO finish writing metrics etc to dataframe