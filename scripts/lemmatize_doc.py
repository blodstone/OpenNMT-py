import os
import json
from stanfordcorenlp import StanfordCoreNLP

nlp = StanfordCoreNLP(r'/home/acp16hh/Projects/Others/stanford-corenlp-full-2018-10-05')
finished_files_dir = "../data/smallbbc-split"


def save_to_json(file_path, save_path):
    file = open(file_path, 'r')
    all_lemmas = []
    line_i = 1
    for line in file:
        print(line_i)
        line_i += 1
        new_line = ' '.join(line.split()[:400])
        parsed_dict = json.loads(nlp.annotate(new_line,
                                              properties={
                                                  'annotators': 'lemma',
                                                  'pipelineLanguage': 'en', 'outputFormat': 'json'}))
        lemmas = ' '.join([token['lemma'].lower()
                               for sentence in parsed_dict['sentences'] for token in sentence['tokens']])
        all_lemmas.append(lemmas)
    file.close()
    f_lemma = open(finished_files_dir + save_path, "w")
    f_lemma.write('\n'.join(all_lemmas).strip())
    f_lemma.close()



print('Processing train file:')
save_to_json('../data/smallbbc-split/src.txt.train', "/src.lemma.train")
print('Processing validation file:')
save_to_json('../data/smallbbc-split/src.txt.validation', "/src.lemma.validation")
print('Processing test file:')
save_to_json('../data/smallbbc-split/src.txt.test', "/src.lemma.test")
