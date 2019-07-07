import argparse
import logging
import string
import spacy
import re
import os
import joblib
import gensim
import pickle

sp = spacy.load('en_core_web_sm',
                disable=['ner', 'parser', 'textcat', 'entity_ruler', 'merge_noun_chunks',
                         'merge_entities', 'merge_subtokens'])
p = re.compile(r'.*\d.*')

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
    for doc in sp.pipe(lines, batch_size=1000, n_threads=7):
        if i % 1000 == 0:
            logging.log(logging.INFO, 'Parse line {}'.format(i))
        i += 1
        new_line = [token.lemma_.lower() for token in doc
                    if not p.match(token.text)
                    and token.text not in string.punctuation
                    and not token.is_stop == args.remove_stop_words
                    and token.text.lower() not in ["\'\'", "``", "-lsb-", "-rsb-", "-rrb-", "-lrb-", "\'s", "--", "sos", "eos"]
                    and token.lemma_.lower() != '-pron-']
        doc_list.append(new_line)
    id2word = gensim.corpora.Dictionary(doc_list)
    pickle.dump(doc_list, open(args.output +'/doc_list.pickle', 'wb'))
    corpus = [id2word.doc2bow(doc) for doc in doc_list]
    pickle.dump(id2word, open(args.output + '/id2word.pickle', 'wb'))
    gensim.corpora.MmCorpus.serialize(args.output + '/corpus.mm', corpus)
