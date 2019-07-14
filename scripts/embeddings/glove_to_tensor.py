import argparse
import pickle
import six
import torch


def load_glove(glove_lines):
    embs = dict()
    for glove_line in glove_lines:
        line = glove_line.split()
        embs[line[0]] = [float(i) for i in line[1:]]
    return embs


def main():
    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file', required=False,
                        help="Embeddings path")
    parser.add_argument('-output_file', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', required=True,
                        help="Dictionary file")
    parser.add_argument('-lemma_align', required=True,
                        help="Pair dictionary")

    opt = parser.parse_args()
    fields = torch.load(opt.dict_file)
    tgt_vocab = fields['src'].base_field.vocab
    glove_file = open(opt.emb_file, 'r')
    embs = load_glove(glove_file.readlines())
    lemma_aligns = open(opt.lemma_align, 'rb').readlines()
    w2l = {}
    for pair in lemma_aligns:
        pair = pair.strip().split()
        word = pair[0].decode('utf-8')
        lemma = pair[1].decode('utf-8')
        if word not in w2l:
            w2l[word] = [lemma]
        else:
            w2l[word].append(lemma)
    dim = len(six.next(six.itervalues(embs)))
    tensor = torch.zeros((len(tgt_vocab), dim))
    empty_word_topic = [0 for i in range(dim)]
    embs['UNK'] = empty_word_topic
    i = 1
    words = []
    for idx, word in enumerate(tgt_vocab.itos):
        if word not in w2l:
            tensor[idx] = torch.tensor(embs['UNK'])
            words.append(word)
            i += 1
        else:
            found = False
            for w2l_word in w2l[word]:
                if w2l_word in embs:
                    tensor[idx] = torch.tensor([float(i) for i in embs[w2l_word]])
                    found = True
            if not found:
                tensor[idx] = torch.tensor((embs['UNK']))
                words.append(word)
                i += 1
    torch.save(tensor, opt.output_file)

if __name__ == '__main__':
    main()
