import re
import json
from stanfordcorenlp import StanfordCoreNLP

def process(path):
    file_r = open(path)
    content = [line.strip().lower() for line in file_r.readlines()]
    file_w = open(path+'.lower', 'w')
    file_w.write('\n'.join(content))
    file_r.close()
    file_w.close()


print('Processing train file:')
# process('../data/bbc-split/src.train.token')
# process('../data/bbc-split/tgt.txt.train')
process('../data/bbc-split/src.train.lemma')

print('Processing validation file:')
# process('../data/bbc-split/src.validation.token')
# process('../data/bbc-split/tgt.txt.validation')
process('../data/bbc-split/src.validation.lemma')

print('Processing test file:')
# process('../data/bbc-split/src.test.token')
# process('../data/bbc-split/tgt.txt.test')
process('../data/bbc-split/src.test.lemma')