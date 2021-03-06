#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""
import codecs
import glob
import sys
import gc
import torch
from functools import partial

from onmt.utils.logging import init_logger, logger
from onmt.utils.misc import split_corpus, split_topic
import onmt.inputters as inputters
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser


def check_existing_pt_files(opt):
    """ Check if there are existing .pt files to avoid overwriting them """
    pattern = opt.save_data + '.{}*.pt'
    for t in ['train', 'valid', 'vocab']:
        path = pattern.format(t)
        if glob.glob(path):
            sys.stderr.write("Please backup existing pt files: %s, "
                             "to avoid overwriting them!\n" % path)
            sys.exit(1)


def build_save_dataset(corpus_type, fields, src_reader, tgt_reader, lemma_reader, topic_reader, opt):
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        src = opt.train_src
        tgt = opt.train_tgt
        topic = opt.train_topic
        lemma = opt.train_lemma
    else:
        src = opt.valid_src
        tgt = opt.valid_tgt
        topic = opt.valid_topic
        lemma = opt.valid_lemma

    logger.info("Reading source, target and topic files: %s %s %s." % (src, tgt, topic))

    src_shards = split_corpus(src, opt.shard_size)
    tgt_shards = split_corpus(tgt, opt.shard_size)
    lemma_shards = split_corpus(lemma, opt.shard_size)
    topic_shards = split_topic(topic, opt.shard_size)

    shard_pairs = zip(src_shards, tgt_shards, topic_shards, lemma_shards)
    dataset_paths = []
    if (corpus_type == "train" or opt.filter_valid) and tgt is not None:
        filter_pred = partial(
            inputters.filter_example, use_src_len=opt.data_type == "text",
            max_src_len=opt.src_seq_length, max_tgt_len=opt.tgt_seq_length)
    else:
        filter_pred = None

    for i, (src_shard, tgt_shard, topic_shard, lemma_shards) in enumerate(shard_pairs):
        if (len(src_shard) != len(tgt_shard)):
            print("Src: {}\n Tgt: {}".format(len(src_shard), len(tgt_shard)))
        assert len(src_shard) == len(tgt_shard)
        if (len(src_shard) != len(topic_shard)):
            print("Src: {}\n Topic: {}".format(len(src_shard), len(topic_shard)))
        assert len(src_shard) == len(topic_shard)
        logger.info("Building shard %d." % i)
        dataset = inputters.Dataset(
            fields,
            readers=[src_reader, tgt_reader, topic_reader, lemma_reader]
                if tgt_reader else [src_reader, topic_reader, lemma_reader],
            data=([("src", src_shard), ("tgt", tgt_shard), ("doc_topic", topic_shard), ("word_topic", lemma_shards)]
                  if tgt_reader else [("src", src_shard), ("doc_topic", topic_shard), ("word_topic", lemma_shards)]),
            dirs=[opt.src_dir, None, None, None] if tgt_reader else [opt.src_dir, None, None],
            sort_key=inputters.str2sortkey[opt.data_type],
            filter_pred=filter_pred
        )

        data_path = "{:s}.{:s}.{:d}.pt".format(opt.save_data, corpus_type, i)
        dataset_paths.append(data_path)

        logger.info(" * saving %sth %s data shard to %s."
                    % (i, corpus_type, data_path))

        dataset.save(data_path)

        del dataset.examples
        gc.collect()
        del dataset
        gc.collect()

    return dataset_paths


def build_save_vocab(train_dataset, fields, opt):
    fields = inputters.build_vocab(
        train_dataset, fields, opt.data_type, opt.share_vocab,
        opt.src_vocab, opt.src_vocab_size, opt.src_words_min_frequency,
        opt.tgt_vocab, opt.tgt_vocab_size, opt.tgt_words_min_frequency,
        opt.lemma_vocab, opt.lemma_vocab_size, opt.lemma_words_min_frequency,
        vocab_size_multiple=opt.vocab_size_multiple
    )

    vocab_path = opt.save_data + '.vocab.pt'
    torch.save(fields, vocab_path)


def count_features(path):
    """
    path: location of a corpus file with whitespace-delimited tokens and
                    ￨-delimited features within the token
    returns: the number of features in the dataset
    """
    with codecs.open(path, "r", "utf-8") as f:
        first_tok = f.readline().split(None, 1)[0]
        return len(first_tok.split(u"￨")) - 1


def main(opt):
    ArgumentParser.validate_preprocess_args(opt)

    torch.manual_seed(opt.seed)
    check_existing_pt_files(opt)

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    src_nfeats = count_features(opt.train_src) if opt.data_type == 'text' \
        else 0
    tgt_nfeats = count_features(opt.train_tgt)  # tgt always text so far
    logger.info(" * number of source features: %d." % src_nfeats)
    logger.info(" * number of target features: %d." % tgt_nfeats)

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(
        opt.data_type,
        src_nfeats,
        tgt_nfeats,
        lower=opt.lower,
        dynamic_dict=opt.dynamic_dict,
        src_truncate=opt.src_seq_length_trunc,
        src_sent_truncate=opt.src_sent_length_trunc,
        tgt_truncate=opt.tgt_seq_length_trunc,
    )

    src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
    tgt_reader = inputters.str2reader["text"].from_opt(opt)
    doc_topic_reader = inputters.str2reader["doc_topic"].from_opt(opt)
    word_topic_reader = inputters.str2reader["text"].from_opt(opt)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset(
        'train', fields, src_reader, tgt_reader, word_topic_reader, doc_topic_reader, opt)

    if opt.valid_src and opt.valid_tgt:
        logger.info("Building & saving validation data...")
        build_save_dataset('valid', fields, src_reader, tgt_reader, word_topic_reader, doc_topic_reader, opt)

    logger.info("Building & saving vocabulary...")
    build_save_vocab(train_dataset_files, fields, opt)


def _get_parser():
    parser = ArgumentParser(description='preprocess.py')

    opts.config_opts(parser)
    opts.preprocess_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
