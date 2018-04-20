from __future__ import absolute_import
import os
import random
from collections import defaultdict
import pickle
import logging
from tqdm import tqdm

from Averageperceptron import AveragedPerceptron

PICKLE = "data/trontagger-0.1.0.pickle"

class PerceptronTagger():

    START = ['-START-','-START2-']
    END = ['-END-','-END2']
    AP_MODEL_LOC = os.path.join(os.path.dirname(__file__),PICKLE)

    def __init__(self,load=True):
        self.model = AveragedPerceptron()
        self.tagdict = {}
        self.classes = set()
        if load:
            self.load(self.AP_MODEL_LOC)

    def tag(self,corpus):
        s_split = lambda t: t.split('\n')
        w_split = lambda s: s.split()

        def split_sents(corpus):
            for s in s_split(corpus):
                yield w_split(s)

        prev,prev2 = self.START
        tokens = []
        for words in split_sents(corpus):
            context = self.START + [self._normalize(w) for w in words] + self.END
            for i,word in enumerate(words):
                tag = self.tagdict.get(word)
                if not tag:
                    features = self._get_features(i,word,context,prev,prev2)
                    tag = self.model.predict(features)
                tokens.append((word,tag))
                prev2 = prev
                prev = tag
        return tokens

    def train(self,sentences,save_loc = None,nr_iter=5):
        self._make_tagdict(sentences)
        self.model.classes = self.classes
        for iter in tqdm(range(nr_iter)):
            c = 0
            n = 0
            '''n为每一轮训练总数'''
            for words,tags in tqdm(sentences):
                prev,prev2 = self.START
                contex = self.START + [self._normalize(w) for w in words] + self.END
                for i,word in enumerate(words):
                    '''i为word的索引数'''
                    guess = self.tagdict.get(word)
                    '''取word对应的tag,为guess'''
                    if not guess:
                        feats = self._get_features(i,word,contex,prev,prev2)
                        guess = self.model.predict(feats)
                        self.model.update(tags[i],guess,feats)
                    prev2 = prev
                    prev = guess
                    c += guess ==tags[i]
                    n += 1
            random.shuffle(sentences)
            logger.info("Iter {0}:{1}/{2}={3}".format(iter,c,n,self._pc(c,n)))
        self.model.average_weights()
        if save_loc is not None:
            pickle.dump((self.model.weights,self.tagdict,self.classes),open(save_loc,'wb'),-1)
        return None

    def _pc(self,n,d):
        return (float(n)/d) *100


    def load(self,loc):
        try:
            w_td_c = pickle.load(open(loc,'rb'))
        except IOError:
            msg = ("Missing trontagger.pickle file.")
            raise IOError(msg)
        self.model.weights, self.tagdict, self.classes = w_td_c
        self.model.classes = self.classes
        return None

    def _normalize(self,word):
        if '-' in word and word[0] != '-':
            return '!HYPHEN'
        elif word.isdigit() and len(word) == 4:
            return '!YEAR'
        elif word[0].isdigit():
            return '!DIGITS'
        else:
            return word.lower()

    def _get_features(self,i,word,context,prev,prev2):

        def add(name, *args):
            features[' '.join((name,) + tuple(args))] += 1

        i += len(self.START)
        features = defaultdict(int)

        add('bias')
        add('i suffix',word[-3:])
        add('i pref1', word[0])
        add('i-1 tag',prev)
        add('i-2 tag',prev2)
        add('i-1 tag+i-2 tag',prev,prev2)
        add('i word',context[i])
        add('i-1 tag+i word',prev,context[i])
        add('i-1 word',context[i-1])
        add('i-1 suffix',context[i-1][-3:])
        add('i-2 word',context[i-2])
        add('i+1 word',context[i+1])
        add('i+1 suffix',context[i+1][-3:])
        add('i+2 word',context[i+2])
        return features

    def _make_tagdict(self,sentences):
        counts = defaultdict(lambda: defaultdict(int))
        '''count是dict的dict'''
        for words,tags in sentences:
            for word,tag in zip(words,tags):
                counts[word][tag] += 1
                self.classes.add(tag)
                '''self.classes是set，只能包含不相同的元素'''
        freq_thresh = 20
        ambiguity_thresh = 0.97
        for word, tag_freqs in counts.items():
            tag,mode = max(tag_freqs.items(),key = lambda item:item[1])
            n = sum(tag_freqs.values())
            if n >= freq_thresh and (float(mode)/n) >= ambiguity_thresh:
                self.tagdict[word] = tag


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler("log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    tagger = PerceptronTagger(False)
    try:
        tagger.load(PICKLE)
        print(tagger.tag('how are you ?'))
        logger.info('Start testing...')
        right =0.0
        total =0.0
        sentence = ([],[])
        for line in tqdm(open('data/check.txt')):
            params = line.split()
            if len(params)!=2:continue
            sentence[0].append(params[0])
            sentence[1].append(params[1])
            if params[0] == '.':
                test = ''
                words = sentence[0]
                tags = sentence[1]
                for i,word in enumerate(words):
                    test += word
                    if i<len(words):test += ' '
                outputs = tagger.tag(test)
                assert len(tags) == len(outputs)
                total += len(tags)
                for o,t in zip(outputs,tags):
                    if o[1].strip() == t:right +=1
                sentence = ([],[])
        logger.info("Precision:%f",right/total)
    except IOError:
        logger.info('Reading corpus...')
        training_data = []
        sentence = ([],[])
        '''tuple'''
        for line in open('data/train.txt'):
            params = line.split()
            sentence[0].append(params[0])
            sentence[1].append(params[1])
            if params[0] == '.':
                training_data.append(sentence)
                sentence = ([],[])
        logger.info('training corpus size: %d',len(training_data))
        logger.info('Start training...')
        tagger.train(training_data,save_loc=PICKLE)
