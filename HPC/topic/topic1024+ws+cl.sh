#!/usr/bin/env bash
#$ -l gpu=1
#$ -P rse
#$ -q rse.q
#$ -l rmem=8G
#$ -l h_rt=24:00:00
#$ -M hhardy2@sheffield.ac.uk
#$ -m easb
#$ -wd /home/acp16hh/Topic_Summ

WORK=/home/acp16hh/Topic_Summ
DATA=/data/acp16hh/bbc-split
MODEL=/data/acp16hh/models
OUTPUT=/data/acp16hh/output
module load apps/python/conda
module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176
source activate freya

#python preprocess.py -train_src $DATA/src.train.token -train_topic $DATA/src.lda.train -train_lemma $DATA/src.train.lemma -train_tgt $DATA/tgt.txt.train -valid_src $DATA/src.validation.token -valid_tgt $DATA/tgt.txt.validation -valid_topic $DATA/src.lda.validation -valid_lemma $DATA/src.validation.lemma -save_data $DATA/topic-bbc-split -src_seq_length 10000 -tgt_seq_length 10000 -src_seq_length_trunc 400 -tgt_seq_length_trunc 100 -dynamic_dict -share_vocab -shard_size 100000

python train.py -save_model $MODEL/topic-bbc-1024-ws-cl -data $DATA -global_attention mlp -word_vec_size 256 -rnn_size 1024 -layers 2 -encoder_type brnn -train_steps 200000 -max_grad_norm 2 -dropout 0. -batch_size 16 -valid_batch_size 16 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -seed 777 -sample weighted -fusion average -topic_psi 0.25 -topic_matrix $DATA/topic_matrix.tensor -lemma-align $DATA/src.train.pair -bridge -world_size 1 -gpu_ranks 0 -log_file $OUTPUT/topic-bbc-1024-ws-cl-01.log

#python translate.py -gpu 0 -batch_size 20 -beam_size 10 -model models/bbc-split_step_10000.pt -src data/bbc-split/src.test.token -lemma data/bbc-split/src.test.lemma -lemma-align data/bbc-split/src.train.pair -topic_matrix data/bbc-split/topic_matrix.tensor -output testout/smallbbc.out -min_length 35 -verbose -stepwise_penalty -coverage_penalty summary -beta 5 -length_penalty wu -alpha 0.9 -block_ngram_repeat 3 -ignore_when_blocking "." "</t>" "<t>" "<sos>" "<eos>" --report_rouge --replace_unk -log_file testout/translate_see_01.log
