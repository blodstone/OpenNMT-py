import argparse
import logging
import string
import spacy
import re
import os
import random
import gensim
import pickle

sp = spacy.load('en_core_web_sm',
                disable=['ner', 'parser', 'textcat', 'entity_ruler', 'sentencizer', 'merge_noun_chunks',
                         'merge_entities', 'merge_subtokens'])
p = re.compile(r'.*\d.*')

if __name__ == '__main__':
    program = os.path.basename("Create Topic Corpus")
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-bbcdocs', help='The input BBC documents')
    parser.add_argument('-cnndmdocs', help='The input CNN/DM documents')
    parser.add_argument('-output', help="The path for preprocessing output")
    args = parser.parse_args()
    doc_list = []
    bbc_lines = open(args.bbcdocs).readlines()
    cnndm_lines = open(args.cnndmdocs).readlines()
    lines = bbc_lines + cnndm_lines
    random.shuffle(lines)
    logging.log(logging.INFO, 'Total line {}'.format(len(lines)))
    i = 0
    for doc in sp.pipe(bbc_lines + cnndm_lines, batch_size=100):
        if i % 100 == 0:
            logging.log(logging.INFO, 'Parse line {}'.format(i))
        i += 1
        new_line = [token.lemma_.lower() for token in doc
                    if not p.match(token.text)
                    and token.text not in string.punctuation
                    and token.text.lower() not in ["\'\'", "``", "-rrb-", "-lrb-", "-llb-", "-rlb-", "\'s", "--", "sos", "eos"]
                    and token.lemma_.lower() != '-pron-'
                    and len(token.text) > 2]
        doc_list.append(new_line)
    id2word = gensim.corpora.Dictionary(doc_list)
    corpus = [id2word.doc2bow(doc) for doc in doc_list]
    pickle.dump(id2word, open(args.output + '/id2word.pickle', 'wb'))
    gensim.corpora.MmCorpus.serialize(args.output + '/corpus.mm', corpus)
