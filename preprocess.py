import pandas as pd
import sys
import re
from urllib.request import urlopen
from tqdm._tqdm_notebook import tqdm_notebook
tqdm_notebook.pandas()
import copy
from string import punctuation
from gensim.parsing.preprocessing import STOPWORDS
from collections import Counter
class MyTokenizer():
    def __init__(self,remove_markers=False,remove_stopwords=False,lower=True,remove_punct=True):
        FLAGS = re.MULTILINE | re.DOTALL
        self.FLAGS=FLAGS
        #word_url='https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt'
        word_url='https://gist.githubusercontent.com/h3xx/1976236/raw/bbabb412261386673eff521dddbe1dc815373b1d/wiki-100k.txt'
        dictionary_data =urlopen(word_url).read()
        dictionary=[word.decode('utf-8') for word in dictionary_data.split()]
        dictionary=[word for word in dictionary if len(word)>1]
        dictionary.extend(['shit','bullshit','islam','racist','sexist','islamic','sexism','racism','feminism',
                      'feminist','fucks','fuck','twat'])
        self.dictionary=dictionary
        self.remove_markers=remove_markers
        self.remove_stopwords=remove_stopwords
        self.lower=lower
        self.remove_punct=remove_punct

    
    def get_word(self,token):
        i = len(token) + 1
        while i > 1:
            i -= 1
            if token[:i] in self.dictionary:
                return token[:i]
        return None 

    def parse_hashtag(self,term):
        words = []
        # Remove hashtag, split by dash
        tags = term.split('-')
        for tag in tags:
            word = self.get_word(tag)    
            while word != None and len(tag) > 0:
                words.append(word)            
                if len(tag) == len(word): 
                    break
                tag = tag[len(word):]
                word = self.get_word(tag)
        return ' '.join(words)

    @staticmethod
    def split_on_caps(text):
        words=re.findall('[A-Z][^A-Z]*',text)
        return ' '.join(words)

    def hashtag(self,text):
        text = text.group()
        tag = text[1:]
        if tag.isupper():
            TAG= tag
        elif tag.islower():
            if tag not in self.dictionary:
                TAG = self.parse_hashtag(tag)
            else:
                TAG = tag
        else:
            if tag.lower() not in self.dictionary:
                TAG = self.split_on_caps(tag)
            else:
                TAG = tag
        if self.remove_markers==True:
            return TAG
        else:
            return " <hashtag> "+TAG

    def allcaps(self,text):
        text = text.group()
        if self.remove_markers==True:
            return text #+ " <allcaps>"
        else:
            return text + " <allcaps> "
        
    def twitter_preprocess(self,text):
        # Different regex parts for smiley faces
        eyes = r"[8:=;]"
        nose = r"['`\-]?"

        #global remove
        #remove=remove_markers

        # function so code less repetitive
        def re_sub(pattern, repl):
            return re.sub(pattern, repl, text, flags=self.FLAGS)

        if self.remove_markers==False:
            text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", " <url> ")
            text = re_sub(r"/"," / ")
            text = re_sub(r"@\w+", " <user> ")
            text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), " <smileface> ")
            text = re_sub(r"{}{}p+".format(eyes, nose), " <lolface> ")
            text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), " <sadface> ")
            text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), " <neutralface> ")
            text = re_sub(r"<3"," <heart> ")
            text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <number> ")
            text = re_sub(r"#\S+", self.hashtag)
            text = re_sub(r"([!?.]){2,}", r"\1  <repeat> ")
            text = re_sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2  <elong> ")
            text = re_sub(r"([A-Z]){2,}", self.allcaps)
        else:
            text = re_sub(r"https?:\/\/\S+\b|www\.(\w+\.)+\S*", "")
            text = re_sub(r"/"," / ")
            text = re_sub(r"@\w+", "")
            text = re_sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "")
            text = re_sub(r"{}{}p+".format(eyes, nose), "")
            text = re_sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "")
            text = re_sub(r"{}{}[\/|l*]".format(eyes, nose), "")
            text = re_sub(r"<3","")
            text = re_sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "")
            text = re_sub(r"#\S+", self.hashtag)
            text = re_sub(r"([!?.]){2,}", r"\1")
            text = re_sub(r"\b(\S*?)(.)\2{2,}\b",r"\1\2")
            text = re_sub(r"([A-Z]){2,}", self.allcaps)
        return text

    @staticmethod
    def is_marker(text):
        b="<allcaps>|<hashtag>|<lolface>|<sadface>|<smileface>|<heart>|<number>|<repeat>|<elong>|<neutralface>|<url>|<user>"
        markers = re.findall(b, text)
        if markers:
            return True
        else:
            return False

    def strip_punctuation(self,text):
        marker_punct=["<",">"]
        marker_punctuation=[p for p in punctuation if p not in marker_punct]
        markers=["<allcaps>",
                 "<hashtag>",
                 "<lolface>",
                 "<sadface>",
                 "<smileface>",
                 "<heart>",
                 "<number>",
                 "<repeat>",
                 "<elong>",
                 "<neutralface>",
                 "<url>",
                 "<user>"]
        tokens=text.split()
        stripped_tokens=[]
        for token in tokens:
            token=token.strip()
            if self.remove_markers==True:
                 stripped_tokens.append(''.join([p for p in token if p not in punctuation]))

            if self.remove_markers==False:
                if not self.is_marker(token):
                    stripped_tokens.append(''.join([p for p in token if p not in punctuation]))
                else:
                    stripped_tokens.append(''.join([p for p in token if p not in marker_punctuation]))

        return ' '.join(stripped_tokens)

    
    def preprocess_text(self,text):
        text=self.twitter_preprocess(text)
        if self.remove_punct==True:
            text=self.strip_punctuation(text)
        tokens = text.split()
        if self.remove_stopwords:
            tokens = [token for token in tokens if token.lower() not in STOPWORDS]
        if self.lower:
            tokens = [token.lower() for token in tokens]
        return " ".join(tokens)


def preprocess_df(df,tokenizer,x_cols):
    ndf=copy.copy(df)
    for col in x_cols:
        print('processsing ',col,' text...')
        ndf[col]=ndf[col].progress_apply(tokenizer)
    #ndf=ndf.drop(labels=x_cols,axis=1)
    return ndf

def get_corpus(df,x_cols):
    corpus=[df[col] for col in x_cols]
    corpus=[text for col in corpus for text in col]
    corpus=[text for text in corpus if type(text)==str] #added
    corpus=[word for text in corpus for word in text.split()]
    return corpus

def get_vocab(corpus):
    return list(set(corpus))

def get_uncommon_words(corpus,thresh=4):
    word_counts=Counter(corpus).most_common()[::-1]
    i=0
    for word, count in word_counts:
        i+=1
        if count>thresh:
            break
    print(i-1,' words have frequency < ',thresh, ' and have been removed')
    return [word for word,count in word_counts[:i-1]]


def remove_uncommon_words_from_df(df,x_cols,uncommon_list=[]):
    adf=copy.copy(df)
    global uncommon
    uncommon=uncommon_list
    
    def remove_uncommon_words(text):
        return ' '.join([word for word in text.split() if word not in uncommon])

    for col in x_cols:
        print('removing uncommon words from ',col,' text...')
        adf[col]=adf[col].progress_apply(remove_uncommon_words)
    return adf


def is_in(word,adict):
    try:
        adict[word]
        return True
    
    except KeyError:
        return False




def keep_common_words_in_df(df,x_cols,common_list=[]):
    adf=copy.copy(df)
    com_dict={word:0 for word in common_list}
    def keep_common_words(text):
        return ' '.join([word for word in text.split() if is_in(word,com_dict)])

    for col in x_cols:
        adf[col]=adf[col].progress_apply(keep_common_words)
    return adf