import logging
import os
import argparse
import pickle
import gensim



if __name__ == '__main__':
    program = os.path.basename("Create Topic Model")
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-corpus', help='The input corpus')
    parser.add_argument('-id2word', help='The input dictionary')
    parser.add_argument('-output', help="The path for preprocessing output")
    parser.add_argument('-topic', help="Topic number", type=int)
    args = parser.parse_args()

    TOTAL_NUM_TOPICS = args.topic

    if not os.path.isdir(os.path.dirname(args.output)):
        raise SystemExit("Error: The output directory does not exist. Create the directory and try again.")
    id2word = pickle.load(open(args.id2word, 'rb'))
    mm = gensim.corpora.MmCorpus(args.corpus)
    lda = gensim.models.ldamulticore.LdaModel(corpus=mm, id2word=id2word, num_topics=TOTAL_NUM_TOPICS, chunksize=10000, alpha='asymmetric', update_every=1, passes=1, random_state=100, minimum_probability=None, minimum_phi_value=None, per_word_topics=True)
    lda.save(args.output + '/lda.model')

