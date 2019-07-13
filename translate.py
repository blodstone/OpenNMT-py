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

    if opt.gpu >= 0:
        topic_matrix = torch.load(opt.topic_matrix,
                                  map_location=torch.device(opt.gpu))
    else:
        topic_matrix = torch.load(opt.topic_matrix)
    topic = dict()
    topic['topic_matrix'] = topic_matrix
    topic['topic_joint_attn_mode'] = opt.joint_attn_mode
    if opt.joint_attn_mode == 'co_attention':
        topic['pooling'] = opt.pooling
        topic['weighted_co_attn'] = opt.weighted_co_attn
    else:
        topic['theta'] = opt.theta
    topic['topic_attn_type'] = opt.topic_attn
    topic['topic_attn_func'] = opt.topic_attn_function
    topic['replace_unk_topic'] = opt.replace_unk_topic

    translator = build_translator(opt, topic, report_score=True)
    src_shards = split_corpus(opt.src, opt.shard_size)
    tgt_shards = split_corpus(opt.tgt, opt.shard_size) \
        if opt.tgt is not None else repeat(None)
    shard_pairs = zip(src_shards, tgt_shards)

    for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate(
            topic=topic,
            src=src_shard,
            tgt=tgt_shard,
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
