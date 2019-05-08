import re
import json
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'/home/acp16hh/Projects/Others/stanford-corenlp-full-2018-10-05')
finished_files_dir = "../data/bbc-split"


def parse_sentences(sentences):
    joined_sentence = ''
    for sentence in sentences:
        sentence = '<sos> {} <eos>'.format(sentence)
        joined_sentence = '{} {}'.format(joined_sentence, sentence).strip()
    parsed_dict = json.loads(
        nlp.annotate(joined_sentence,
                     properties={
                         'annotators': 'tokenize,lemma',
                         'pipelineLanguage': 'en',
                         'outputFormat': 'json'}))
    # (word, lemma)
    return ([token['word']
            for sentence in parsed_dict['sentences']
            for token in sentence['tokens']], [token['lemma']
            for sentence in parsed_dict['sentences']
            for token in sentence['tokens']])


def save_to_json(file_path, save_path, is_save=True):
    file = open(file_path, 'r')
    all_lemmas = []
    all_tokens = []
    all_pairs = []
    line_i = 1
    for line in file:
        print(line_i)
        line_i += 1
        sentences = re.findall("<sos> (.*?) <eos>", line)
        split_size = int(len(sentences) / 2)
        parsed_1 = parse_sentences(sentences[:split_size])
        parsed_2 = parse_sentences(sentences[split_size:])
        tokens = parsed_1[0] + parsed_2[0]
        lemmas = parsed_1[1] + parsed_2[1]
        all_pairs.extend(set(zip(tokens, lemmas)))
        all_lemmas.append(' '.join(lemmas).strip())
        all_tokens.append(' '.join(tokens).strip())
    file.close()
    if is_save:
        f_lemma = open(finished_files_dir + save_path + '.lemma', "w")
        f_lemma.write('\n'.join(all_lemmas).strip())
        f_lemma.close()
        f_lemma = open(finished_files_dir + save_path + '.token', "w")
        f_lemma.write('\n'.join(all_tokens).strip())
        f_lemma.close()
    return all_pairs

print('Processing train file:')
all_pairs_src = save_to_json('../data/bbc-split/src.txt.train', "/src.train")
all_pairs_tgt = save_to_json('../data/bbc-split/tgt.txt.train', "", False)
all_pairs = '\n'.join(['{} {}'.format(pair[0], pair[1])
                      for pair in set(all_pairs_src + all_pairs_tgt)])
f_lemma = open(finished_files_dir + '/src.train.pair', "w")
f_lemma.write(all_pairs)
print('Processing validation file:')
save_to_json('../data/bbc-split/src.txt.validation', "/src.validation")
print('Processing test file:')
save_to_json('../data/bbc-split/src.txt.test', "/src.test")
