import argparse
import logging
import string
import spacy
import re
import os
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import gensim
import pickle

# sp = spacy.load('en_core_web_sm',
#                 disable=['ner', 'parser', 'textcat', 'entity_ruler', 'merge_noun_chunks',
#                          'merge_entities', 'merge_subtokens'])
p = re.compile(r'.*\d.*')
stopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

if __name__ == '__main__':
    program = os.path.basename("Create Topic Corpus")
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-docs', help='The input documents')
    parser.add_argument('-output', help="The path for preprocessing output")
    parser.add_argument('--remove_stop_words', help="Remove stop words", action='store_true')
    args = parser.parse_args()
    doc_list = []
    lines = open(args.docs).readlines()
    i = 0
    for doc in lines:
        if i % 1000 == 0:
            logging.log(logging.INFO, 'Parse line {}'.format(i))
        i += 1
        pos_doc = nltk.pos_tag(doc.split( ))
        new_line = [lemmatizer.lemmatize(token, get_wordnet_pos(pos)).lower() for token, pos in pos_doc
                    if not p.match(token)
                    and token not in string.punctuation
                    and (token not in stopWords) == args.remove_stop_words
                    and token.lower() not in ["\'\'", "``", "-lsb-", "-rsb-", "-rrb-", "-lrb-", "\'s", "--", "<sos>", "<eos>"]]
        # print(i)
        # if 'on' in new_line:
        #     print(new_line)
        doc_list.append(new_line)
    id2word = gensim.corpora.Dictionary(doc_list)
    pickle.dump(doc_list, open(args.output +'/doc_list.pickle', 'wb'))
    corpus = [id2word.doc2bow(doc) for doc in doc_list]
    pickle.dump(id2word, open(args.output + '/id2word.pickle', 'wb'))
    gensim.corpora.MmCorpus.serialize(args.output + '/corpus.mm', corpus)
