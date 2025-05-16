import json
import ast
import spacy
import nltk
nltk.download('punkt')
nltk.download('punk_tab')
from spacy.tokens import Doc
from transformers import BertTokenizer, AutoTokenizer
from nlp_id.postag import PosTag
from nltk.tokenize import word_tokenize

#import tokenizations
import os
from spacy_alignments import get_alignments
from transformers.models.bert.tokenization_bert import whitespace_tokenize

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # all tokens have a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)


class DataManager:
    def __init__(self, path, testfile, test):
        # POS tagger
        # nlp = spacy.load("en_core_web_sm")
        # nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)
        #nltk.download('punk_tab')
        #nlp = spacy.load("id_core_news_sm")

        self.postagger = PosTag()
        
        # BERT tokeniser
        # bert_tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
        
        # process data
        # NOTE: order for self.all_pos_tags depends on dataset, order has to be the same between dataset splits
        self.all_pos_tags = []
        self.data = {}
        if not test:
            names = ["train", "dev", "test"]
        else:
            names = ["test"]
        for name in names:
            self.data[name] = []
            filename = testfile if name == "test" else name + "_triplets.txt"
            with open(os.path.join(path, filename)) as fl:
                for line in fl.readlines():
                    # process ASTE data for HRL
                    sentence, triplets = line.strip().split('####')
                    # tokenize for BERT
                    # whitespace_tokens = sentence.split()
                    words = word_tokenize(sentence)
                    pos_tags = self.postagger.get_pos_tag(words)

                    bert_tokens = bert_tokenizer.tokenize(sentence)
                    # https://github.com/tamuhey/tokenizations

                    # whitespace_to_bert, bert_to_whitespace = get_alignments(whitespace_tokens, bert_tokens)
                    whitespace_to_bert, _ = get_alignments(words, bert_tokens)

                    # get WordPiece POS tags
                    # doc = nlp(sentence)
                    # pos_tags = ([w.pos_ for w in doc])
                    # for pt in pos_tags:
                    #    pt_b = 'B-' + pt
                    #    pt_i = 'I-' + pt
                    #    if pt_b not in self.all_pos_tags:
                    #        self.all_pos_tags.append(pt_b)
                    #    if pt_i not in self.all_pos_tags:
                    #        self.all_pos_tags.append(pt_i)
                    
                    bert_pos_tags = []
                    #for i, pos_tag in enumerate(pos_tags):
                    for i, bert_indices in enumerate(whitespace_to_bert):
                      tag = pos_tags[i][1]
                      for j, bert_idx in enumerate(bert_indices):
                        if j == 0:
                          bert_pos_tags.append(f"B-{tag}")
                        else:
                          bert_pos_tags.append(f"I-{tag}")

                    for tag in bert_pos_tags:
                      if tag not in self.all_pos_tags:
                        self.all_pos_tags.append(tag)
                        #if len(whitespace_to_bert[i]) > 1:
                        #    bert_pos_tags.append('B-' + pos_tag)
                        #    for j in range(len(whitespace_to_bert[i])-1):
                        #        bert_pos_tags.append('I-' + pos_tag)
                        #else:
                        #    bert_pos_tags.append('B-' + pos_tag)

                    if not test:
                        triplets = ast.literal_eval(triplets)
                        all_triplets = []
                        triplet_pos = []
                        for i, triplet in enumerate(triplets):
                            aspect_ids, opinion_ids, sentiment = triplet
                            final_triplet = {}
                            final_triplet['sentpol'] = sentiment
                            final_triplet['aspect'] = ''
                            final_triplet['opinion'] = ''
                            final_triplet['aspect_tags'] = [0 for i in range(len(bert_tokens))]
                            final_triplet['opinion_tags'] = [0 for i in range(len(bert_tokens))]
                            # align tokens between ASTE-Data-V2's whitespace tokenisation and BERT tokenisation
                            for j, aspect_id in enumerate(aspect_ids):
                                bert_aspect_ids = whitespace_to_bert[aspect_id]
                                for k, bert_aspect_id in enumerate(bert_aspect_ids):
                                    if j == 0:
                                        if k == 0:
                                            final_triplet['aspect'] += bert_tokens[bert_aspect_id].lower()
                                            final_triplet['aspect_tags'][bert_aspect_id] = 2
                                        else:
                                            final_triplet['aspect'] += ' ' + bert_tokens[bert_aspect_id].lower()
                                            final_triplet['aspect_tags'][bert_aspect_id] = 1
                                    else:
                                        if k == 0:
                                            final_triplet['aspect'] += ' ' + bert_tokens[bert_aspect_id].lower()
                                            final_triplet['aspect_tags'][bert_aspect_id] = 1
                                        else:
                                            final_triplet['aspect'] += ' ' + bert_tokens[bert_aspect_id].lower()
                                            final_triplet['aspect_tags'][bert_aspect_id] = 1
                            bert_aspect_tail_id = bert_aspect_id
                            for j, opinion_id in enumerate(opinion_ids):
                                bert_opinion_ids = whitespace_to_bert[opinion_id]
                                for k, bert_opinion_id in enumerate(bert_opinion_ids):
                                    if j == 0:
                                        if k == 0:
                                            final_triplet['opinion'] += bert_tokens[bert_opinion_id].lower()
                                            final_triplet['opinion_tags'][bert_opinion_id] = 2
                                        else:
                                            final_triplet['opinion'] += ' ' + bert_tokens[bert_opinion_id].lower()
                                            final_triplet['opinion_tags'][bert_opinion_id] = 1
                                    else:
                                        if k == 0:
                                            final_triplet['opinion'] += ' ' + bert_tokens[bert_opinion_id].lower()
                                            final_triplet['opinion_tags'][bert_opinion_id] = 1
                                        else:
                                            final_triplet['opinion'] += ' ' + bert_tokens[bert_opinion_id].lower()
                                            final_triplet['opinion_tags'][bert_opinion_id] = 1
                            bert_opinion_tail_id = bert_opinion_id
                            all_triplets.append(final_triplet)
                            triplet_pos.append([i, max(bert_aspect_tail_id, bert_opinion_tail_id)])
                        # sort by max of aspect and opinion span tail positions
                        sorted_triplets = []
                        triplet_pos = sorted(triplet_pos, key=lambda x: (x[1]))
                        for i in triplet_pos:
                            sorted_triplets.append(all_triplets[i[0]])
                        self.data[name].append({'sentext': sentence, 'triplets': sorted_triplets, 'pos_tags': bert_pos_tags, 'bert_to_whitespace': bert_to_whitespace, 'whitespace_tokens': whitespace_tokens})
                    else:
                        # triplets are not needed for inference
                        self.data[name].append({'sentext': sentence, 'triplets': None, 'pos_tags': bert_pos_tags, 'bert_to_whitespace': bert_to_whitespace, 'whitespace_tokens': whitespace_tokens})
                fl.close()

        # convert POS tags to IDs
        for name in names:
            for item in self.data[name]:
                item['pos_tags'] = ([self.all_pos_tags.index(j) for j in item['pos_tags']])

        if not test:
            # process sentiment polarity
            self.sentcnt = {}
            # NOTE: order for self.sentiments depends on dataset, order has to be the same between dataset splits
            self.sentiments = []
            for name in ['train','dev']:
                for item in self.data[name]:
                    for t in item['triplets']:
                        sentpol = t['sentpol']
                        if not sentpol in self.sentiments:
                            self.sentiments.append(sentpol)
                            self.sentcnt[sentpol] = 1
                        else:
                            self.sentcnt[sentpol] += 1
            self.sent_count = len(self.sentiments)
            for name in names:
                for item in self.data[name]:
                    for t in item['triplets']:
                        t['type'] = self.sentiments.index(t['sentpol']) + 1
            print(self.sentcnt)
            print(self.sentiments)
        else:
            # NOTE: order for self.sentiments depends on dataset, order has to be the same between dataset splits
            # fixed self.sentiments order for test data
            self.sentiments = ['POS', 'NEG', 'NEU']
            self.sent_count = len(self.sentiments)
            print(self.sentiments)
