import gensim
import argparse
import logging

if __name__ == "__main__":
    logger = logging.getLogger('Experiment With Topic Matrix')

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-docs', help='The input documents')
    parser.add_argument('-lda', help="The lda model")
    args = parser.parse_args()

    lda = gensim.models.ldamulticore.LdaMulticore.load(args.lda, mmap='r')
    lines = open(args.docs).readlines()

