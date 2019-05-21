import re
import json
from stanfordcorenlp import StanfordCoreNLP


def remove_sos_eos(file):
    i = 1
    result = []
    for line in file:
        print(i)
        i += 1
        l_result = []
        lines = line.split()
        for l in lines:
            if l == '<sos>' or l == '<eos>':
                continue
            l_result.append(l)
        result.append(' '.join(l_result))
    return result


def process(path):
    file_r = open(path)
    file_w = open(path+'.lower_no_tag', 'w')
    file_w.write('\n'.join(remove_sos_eos(file_r)))
    file_r.close()
    file_w.close()


print('Processing train file:')
process('../data/bbc-split/src.train.token')
# process('../data/bbc-split/tgt.txt.train')

print('Processing validation file:')
process('../data/bbc-split/src.validation.token')
# process('../data/bbc-split/tgt.txt.validation')

print('Processing test file:')
process('../data/bbc-split/src.test.token')
# process('../data/bbc-split/tgt.txt.test')
