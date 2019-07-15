import os

def gen_header():
    header = '#!/usr/bin/env bash\n' \
             + '#$ -l gpu=1\n' \
             + '#$ -P rse\n' \
             + '#$ -q rse.q\n' \
             + '#$ -l rmem=16G\n' \
             + '#$ -l h_rt=96:00:00\n' \
             + '#$ -M hhardy2@sheffield.ac.uk\n' \
             + '#$ -m easb\n' \
             + '#$ -wd /home/acp16hh/Topic_Summ\n' \
             + 'WORK=/home/acp16hh/Topic_Summ\n' \
             + 'DATA=/data/acp16hh/preprocess-bbc-split\n' \
             + 'RAW_DATA=/data/acp16hh/bbc-split\n' \
             + 'MODEL=/fastdata/acp16hh/models\n' \
             + 'OUTPUT=/data/acp16hh/output\n' \
             + 'module load apps/python/conda\n' \
             + 'module load libs/cudnn/7.3.1.20/binary-cuda-9.0.176\n' \
             + 'source activate freya\n'
    return header

if __name__ == '__main__':
    data = '$DATA/bbc_topic'
    topic_matrix = '$RAW_DATA/preprocess-topic-bbc/t-512-shashi/topic_matrix.tensor'
    name = ''

    joint_attn_modes = ['co_attention', 'mix', 'embedded']
    thetas = [0.75, 0.5, 0.25]
    poolings = ['column', 'row', 'exp']
    weighted_co_attns = [True, False]
    topic_attns = ['dot', 'mlp']
    topic_attn_functions = ['sparsemax', 'softmax']
    replace_unk_topics = [True, False]
    script_id = 1
    for joint_attn_mode in joint_attn_modes:
        if joint_attn_mode == 'co_attention':
            for pooling in poolings:
                for topic_attn in topic_attns:
                    for topic_attn_function in topic_attn_functions:
                        for replace_unk_topic in replace_unk_topics:
                            name = '{}_{}_{}_{}_{}'.format(
                                joint_attn_mode, pooling,
                                topic_attn, topic_attn_function, replace_unk_topic)
                            log_file = '$OUTPUT/{}.log'.format(name)
                            save_model = '$MODEL/{}/{}'.format(name, name)
                            if not os.path.exists('/fastdata/acp16hh/models/{}'.format(name)):
                                os.makedirs('/fastdata/acp16hh/models/{}'.format(name))
                            if replace_unk_topic:
                                body = 'python train.py -save_model {} -data {} -copy_attn -global_attention mlp -word_vec_size 256 -rnn_size 1024 -layers 2 -encoder_type brnn -train_steps 200000 -max_grad_norm 2 -dropout 0. -batch_size 8 -valid_batch_size 8 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -seed 777 -topic_matrix {} -bridge -world_size 0 -gpu_ranks 0 -reuse_copy_attn -copy_loss_by_seqlength -joint_attn_mode {} -pooling {} -topic_attn {} -topic_attn_function {} -log_file {} -replace_unk_topic'.format(save_model, data, topic_matrix, log_file, joint_attn_mode, pooling, topic_attn, topic_attn_function)
                            else:
                                body = 'python train.py -save_model {} -data {} -copy_attn -global_attention mlp -word_vec_size 256 -rnn_size 1024 -layers 2 -encoder_type brnn -train_steps 200000 -max_grad_norm 2 -dropout 0. -batch_size 8 -valid_batch_size 8 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -seed 777 -topic_matrix {} -bridge -world_size 0 -gpu_ranks 0 -reuse_copy_attn -copy_loss_by_seqlength -joint_attn_mode {} -pooling {} -topic_attn {} -topic_attn_function {} -log_file {}'.format(save_model, data, topic_matrix, log_file, joint_attn_mode, pooling, topic_attn,topic_attn_function)
                            file_path = os.path.join('../{}_topic_{}.sh'.format(script_id, name))
                            script_id += script_id
                            file = open(file_path, 'w')
                            file.write(gen_header() + body)
                            file.close()
        elif joint_attn_mode == 'mix':
            for theta in thetas:
                for topic_attn in topic_attns:
                    for topic_attn_function in topic_attn_functions:
                        for replace_unk_topic in replace_unk_topics:
                            name = '{}_{}_{}_{}_{}'.format(
                                joint_attn_mode, theta,
                                topic_attn, topic_attn_function, replace_unk_topic)
                            log_file = '$OUTPUT/{}.log'.format(name)
                            save_model = '$MODEL/{}/{}'.format(name, name)
                            if not os.path.exists('/fastdata/acp16hh/models/{}'.format(name)):
                                os.makedirs('/fastdata/acp16hh/models/{}'.format(name))
                            if replace_unk_topic:
                                body = 'python train.py -save_model {} -data {} -copy_attn -global_attention mlp -word_vec_size 256 -rnn_size 1024 -layers 2 -encoder_type brnn -train_steps 200000 -max_grad_norm 2 -dropout 0. -batch_size 8 -valid_batch_size 8 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -seed 777 -topic_matrix {} -bridge -world_size 0 -gpu_ranks 0 -reuse_copy_attn -copy_loss_by_seqlength -joint_attn_mode {} -theta {} -topic_attn {} -topic_attn_function {} -log_file {} -replace_unk_topic'.format(save_model, data, topic_matrix, log_file, joint_attn_mode, theta, topic_attn, topic_attn_function)
                            else:
                                body = 'python train.py -save_model {} -data {} -copy_attn -global_attention mlp -word_vec_size 256 -rnn_size 1024 -layers 2 -encoder_type brnn -train_steps 200000 -max_grad_norm 2 -dropout 0. -batch_size 8 -valid_batch_size 8 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -seed 777 -topic_matrix {} -bridge -world_size 0 -gpu_ranks 0 -reuse_copy_attn -copy_loss_by_seqlength -joint_attn_mode {} -theta {} -topic_attn {} -topic_attn_function {} -log_file {}'.format(save_model, data, topic_matrix, log_file, joint_attn_mode, theta, topic_attn,topic_attn_function)
                            file_path = os.path.join('../{}_topic_{}.sh'.format(script_id, name))
                            script_id += script_id
                            file = open(file_path, 'w')
                            file.write(gen_header() + body)
                            file.close()
        elif joint_attn_mode == 'embedded':
            name = '{}'.format(joint_attn_mode)
            log_file = '$OUTPUT/{}.log'.format(name)
            save_model = '$MODEL/{}/{}'.format(name, name)
            if not os.path.exists('/fastdata/acp16hh/models/{}'.format(name)):
                os.makedirs('/fastdata/acp16hh/models/{}'.format(name))
            body = 'python train.py -save_model {} -data {} -copy_attn -global_attention mlp -word_vec_size 256 -rnn_size 1024 -layers 2 -encoder_type brnn -train_steps 200000 -max_grad_norm 2 -dropout 0. -batch_size 8 -valid_batch_size 8 -optim adagrad -learning_rate 0.15 -adagrad_accumulator_init 0.1 -seed 777 -topic_matrix {} -bridge -world_size 0 -gpu_ranks 0 -reuse_copy_attn -copy_loss_by_seqlength -joint_attn_mode {}'.format(save_model, data, topic_matrix, log_file, joint_attn_mode)
            file_path = os.path.join('../{}_topic_{}.sh'.format(script_id, name))
            script_id += script_id
            file = open(file_path, 'w')
            file.write(gen_header() + body)
            file.close()



