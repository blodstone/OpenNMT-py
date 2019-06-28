import logging
import os
import argparse
import re
import string
import gensim
import spacy

sp = spacy.load('en_core_web_sm', disable=['ner', 'parser', 'textcat', 'entity_ruler', 'sentencizer', 'merge_noun_chunks', 'merge_entities', 'merge_subtokens'])
p = re.compile(r'.*\d.*')

if __name__ == '__main__':
    program = os.path.basename("Create Topic Model")
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-docs', help='The input documents')
    parser.add_argument('-output', help="The path for preprocessing output")
    parser.add_argument('-topic', help="Topic number", type=int)
    args = parser.parse_args()
    TOTAL_NUM_TOPICS = args.topic

    if not os.path.isdir(os.path.dirname(args.output)):
        raise SystemExit("Error: The output directory does not exist. Create the directory and try again.")

    doc_list = []
    with open(args.docs) as training_file:
        i = 0
        for line in training_file:
            logging.log(logging.INFO, 'Parse line {}'.format(i))
            i += 1
            new_line = [token.lemma_.lower()
                 for token in sp(line.strip())
                 if not p.match(token.text)
                 and token.text not in string.punctuation
                 and token.text.lower() not in ["\'\'", "``", "-rrb-", "-lrb-", "\'s", "--", "sos", "eos"]
                 and token.lemma_.lower() != '-pron-'
                 and len(token.text) > 2]
            doc_list.append(new_line)

    # print(doc_list)

    id2word = gensim.corpora.Dictionary(doc_list)
    corpus = [id2word.doc2bow(doc) for doc in doc_list]
    gensim.corpora.MmCorpus.serialize(args.output + '/corpus.mm', corpus)
    mm = gensim.corpora.MmCorpus(args.output + '/corpus.mm')
    lda = gensim.models.ldamulticore.LdaModel(corpus=mm, id2word=id2word, num_topics=TOTAL_NUM_TOPICS, chunksize=1000,
                                        alpha='asymmetric', update_every=1, passes=1, random_state=100,
                                        minimum_probability=None, minimum_phi_value=None, per_word_topics=True)
    lda.save(args.output + '/lda.model')

