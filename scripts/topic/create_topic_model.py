import logging
import os
import argparse
import pickle
import gensim
from gensim.corpora import Dictionary


if __name__ == '__main__':
    program = os.path.basename("Create Topic Model")
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser()
    # parser.add_argument('-corpus', help='The input corpus')
    parser.add_argument('-id2word', help='The input dictionary')
    parser.add_argument('-doc_list', help='The input text')
    parser.add_argument('-output', help="The path for preprocessing output")
    parser.add_argument('-topic', help="Topic number", type=int)
    parser.add_argument('-chunksize', help="Topic chunksize", type=int)
    parser.add_argument('-passes', help="Topic passes", type=int)
    parser.add_argument('-alpha', help="Topic alpha")
    parser.add_argument('-eta', help="Topic eta")

    args = parser.parse_args()

    TOTAL_NUM_TOPICS = args.topic

    if not os.path.isdir(os.path.dirname(args.output)):
        raise SystemExit("Error: The output directory does not exist. Create the directory and try again.")
    id2word = pickle.load(open(args.id2word, 'rb'))
    id2word.filter_extremes(no_below=5, no_above=0.9)
    doc_list = pickle.load(open(args.doc_list, 'rb'))
    mm = [id2word.doc2bow(doc) for doc in doc_list]
    # mm = gensim.corpora.MmCorpus(args.corpus)
    lda = gensim.models.ldamulticore.LdaModel(corpus=mm, id2word=id2word, num_topics=TOTAL_NUM_TOPICS, chunksize=args.chunksize, alpha=args.alpha, eta=args.eta, eval_every=10, update_every=1, passes=args.passes, random_state=100, minimum_probability=None, minimum_phi_value=None, per_word_topics=True)
    lda.save(args.output + '/lda.model')

