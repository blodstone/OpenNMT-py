#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
import torch

def main(opt):
    ArgumentParser.validate_translate_opts(opt)
    logger = init_logger(opt.log_file)

    translator = build_translator(opt, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    lemma_shards = split_corpus(opt.lemma, opt.shard_size)
    shard_pairs = zip(src_shards, tgt_shards, lemma_shards)
    lemma_align = open(opt.lemma_align, 'rb').readlines()
    if opt.gpu >= 0:
        topic_matrix = torch.load(opt.topic_matrix,
                                  map_location=torch.device(opt.gpu))
    else:
        topic_matrix = torch.load(opt.topic_matrix)
    if not opt.fp32:
        topic_matrix = topic_matrix.half()
    for i, (src_shard, tgt_shard, lemma_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            lemma_align=lemma_align,
            topic_matrix=topic_matrix,
            src=src_shard,
            tgt=tgt_shard,
            lemma=lemma_shard,
            src_dir=opt.src_dir,
            batch_size=opt.batch_size,
            attn_debug=opt.attn_debug
            )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser)
    return parser


if __name__ == "__main__":
    parser = _get_parser()

    opt = parser.parse_args()
    main(opt)
