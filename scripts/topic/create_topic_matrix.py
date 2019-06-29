import pickle
import os
import gensim
from stanfordcorenlp import StanfordCoreNLP

#constant
finished_files_dir = "/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/preprocess-topic-bbc/t-512"
lda_word_topics = "/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/preprocess-topic-bbc/t-512/word_term_topics.log"

if __name__=='__main__':
    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)
    print("Reading word_topic vector")
    word_topic_dict = {}
    wordid_word_dict = {}
    min_word_topic_prob = 100
    count = 0
    with open(lda_word_topics) as f:
        for line in f:
            count += 1
            if count > 2:
                ldata = line.split()
                wordid = int(ldata[0])
                word = ldata[1]
                word_topic_dict[word] = {}
                wordid_word_dict[wordid] = word
                for topic_prob in ldata[2:]:
                    topic_prob_data = topic_prob.split(":")
                    word_topic_dict[word][topic_prob_data[0]] = float(topic_prob_data[1])
                    if float(topic_prob_data[1]) < min_word_topic_prob:
                        min_word_topic_prob = float(topic_prob_data[1])
    print("min_word_topic_prob: " + str(min_word_topic_prob))
    # str_min_word_topic_prob = str(min_word_topic_prob)
    topic_list = [str(i) for i in range(512)]
    empty_word_topic = [min_word_topic_prob for i in topic_list]

    wordids = wordid_word_dict.keys()
    topic_vectors = {'UNK': empty_word_topic}

    for wordid in wordids:
        word = wordid_word_dict[wordid]
        topic_vector = [word_topic_dict[word][i] if i in word_topic_dict[word] else min_word_topic_prob for i in
                        topic_list]
        topic_vectors[word] = topic_vector

    lda = gensim.models.ldamulticore.LdaMulticore.load('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/preprocess-topic-bbc/t-512/lda.model', mmap='r')

    def save_to_json(file_path, save_path):
        doc_parsed_dict = {
            'docs': []
        }
        saved_parsed_dict = {'vectors': [], 'topic': []}
        line_i = 1
        file = open(file_path, 'r')
        for line in file:
            print(line_i)
            line_i += 1
            bow = lda.id2word.doc2bow(line.split())
            topics = lda.get_document_topics(bow, minimum_probability=min_word_topic_prob,
                                             minimum_phi_value=0)
            saved_parsed_dict['topic'] = [topic[1] for topic in topics]
            doc_parsed_dict['docs'].append(saved_parsed_dict)

        f_topic = open(finished_files_dir + save_path, "wb")
        pickle.dump(doc_parsed_dict, f_topic)
        f_topic.close()
        file.close()

    # print('Processing test file:')
    # save_to_json('../data/bbc-split/src.txt.test', "/src.lda.test")
    print('Processing train file:')
    save_to_json('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.txt.train', "/src.lda.train")
    print('Processing validation file:')
    save_to_json('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.txt.validation', "/src.lda.validation")
    pickle.dump(topic_vectors, open(finished_files_dir + '/topic_vectors.lda', 'wb'))



