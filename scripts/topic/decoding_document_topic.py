import logging
import os
import argparse
import gensim
import spacy
import re
import string

sp = spacy.load('en_core_web_sm',
                disable=['ner', 'parser', 'textcat', 'entity_ruler', 'merge_noun_chunks',
                         'merge_entities', 'merge_subtokens'])
p = re.compile(r'.*\d.*')

if __name__ == '__main__':
    program = os.path.basename("Decoding Document Topic")
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('-lda', required=False,
                       help="LDA path")
    parser.add_argument('-docs', help='The input document')
    parser.add_argument('-output', help="The path for preprocessing output")
    args = parser.parse_args()
    doc_list = []
    lda = gensim.models.ldamulticore.LdaMulticore.load(args.lda, mmap='r')

    lines = open(args.docs).readlines()
    result_lines = ''
    for doc in sp.pipe(lines, batch_size=1000, n_threads=7):
        new_line = [token.lemma_.lower() for token in doc
                    if not p.match(token.text)
                    and token.text not in string.punctuation
                    and token.text.lower() not in ["\'\'", "``", "-rrb-", "-lrb-", "\'s", "--", "sos", "eos"]
                    and token.lemma_.lower() != '-pron-']
        doc_list.append(new_line)
    for doc in doc_list:
        bow = lda.id2word.doc2bow(doc)
        # print bow
        topics = lda.get_document_topics(bow, per_word_topics=True, minimum_probability=0, minimum_phi_value=0)
        result_topics = topics[0]
        result_topics.sort(key=lambda x: x[1], reverse=True)
        result_lines += str(result_topics[:5]) + '\n'
    f = open(args.output + '/debug_topic.txt', 'w')
    f.writelines(result_lines)
