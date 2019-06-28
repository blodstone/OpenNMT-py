import gensim
import argparse
import spacy

sp = spacy.load('en_core_web_sm')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lda', help='LDA Model')
    parser.add_argument('-topic', help='Number of Topic', type=int)
    parser.add_argument('-corpus', help='Corpus')
    parser.add_argument('-word', help='Number of Words', type=int)
    args = parser.parse_args()
    mm = gensim.corpora.MmCorpus(args.corpus)
    lda = gensim.models.ldamodel.LdaModel.load(args.lda)
    NUM_WORDS = args.word
    # for topic in lda.get_topics():
    #     print(topic)
    topic = lda.state.get_lambda()
    print_result = ''
    size_result = ''
    sizes = []
    stopwords = set([word for word in spacy.lang.en.stop_words.STOP_WORDS])
    for i in range(args.topic):
        topic_ = topic[i]
        topic_ = topic_ / topic_.sum()  # normalize to probability distribution
        bestn = gensim.matutils.argsort(topic_, NUM_WORDS, reverse=True)
        result = [(lda.id2word[id], topic_[id]) for id in bestn if not lda.id2word[id] in stopwords and topic_[id]>0.001]
        print_result += 'Topic ' + str(i) + ':  ' + ' + '.join('%.3f*"%s"' % (v, k) for k, v in result)
        print_result += '\n\n'
        size_result += 'Topic ' + str(i) + ':  ' + str(len(result)) + '\n\n'
        sizes.append(len(result))
    print(sorted(sizes))
