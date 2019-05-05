import argparse
import pickle
import six
import torch


def main():
    parser = argparse.ArgumentParser(description='embeddings_to_torch.py')
    parser.add_argument('-emb_file', required=False,
                        help="Embeddings path")
    parser.add_argument('-output_file', required=True,
                        help="Output file for the prepared data")
    parser.add_argument('-dict_file', required=True,
                        help="Dictionary file")
    opt = parser.parse_args()
    fields = torch.load(opt.dict_file)
    try:
        vocab = fields['word_topic'].base_field.vocab
    except AttributeError:
        vocab = fields['word_topic'].vocab
    with open(opt.emb_file, 'rb') as f:
        embs = pickle.load(f)
    dim = len(six.next(six.itervalues(embs)))
    tensor = torch.zeros((len(vocab), dim))
    for word, values in embs.items():
        tensor[vocab.stoi[word]] = torch.Tensor(values)
    torch.save(tensor, opt.output_file)

if __name__ == '__main__':
    main()
