import argparse
import gensim
import pickle
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-lda', help='LDA Model')
    parser.add_argument('-output', help='Output folder')
    args = parser.parse_args()
    lda = gensim.models.ldamodel.LdaModel.load(args.lda)
    count = 0
    result = []
    print('Total lines: {}'.format(len(lda.id2word)))
    word_topic_dict = {}
    wordid_word_dict = {}
    for id in range(len(lda.id2word)):
        if id % 1000 == 0:
            print(id)
        term_topics = lda.get_term_topics(id, minimum_probability=0)
        if len(term_topics) > 0:
            wordid_word_dict[id] = lda.id2word[id]
            word_topic_dict[lda.id2word[id]] = {}
            for topic_prob_id, topic_prob in term_topics:
                word_topic_dict[lda.id2word[id]][topic_prob_id] = repr(float(topic_prob))
    print(len(word_topic_dict))
    topic_list = [i for i in range(100)]
    empty_word_topic = [0 for i in topic_list]
    wordids = wordid_word_dict.keys()
    topic_vectors = {'UNK': empty_word_topic}
    for wordid in wordids:
        word = wordid_word_dict[wordid]
        topic_vector = [word_topic_dict[word][i]
                        if i in word_topic_dict[word].keys() else '0' for i in
                        topic_list]
        topic_vectors[word] = topic_vector
    pickle.dump(topic_vectors, open(args.output + '/topic_vectors.lda', 'wb'))
    #     print('Processed line {}'.format(id))
    #     result.append('{} {} {}\n'.format(id, lda.id2word[id].encode("UTF-8"), " ".join(
    #         [str(item[0]) + ":" + str(item[1]) for item in lda.get_term_topics(id, minimum_probability=0)])))
    # count += 1
    # with open(args.output + '/word_term_topics.log', 'w') as file:
    #     file.writelines(result)
