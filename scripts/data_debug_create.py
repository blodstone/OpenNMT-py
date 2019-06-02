file_lemma = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.validation.lemma.lower')
file_token = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.validation.token.lower')
file_tgt = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/tgt.txt.validation')

lemma_len = [len(sent.split()) for sent in file_lemma.readlines()]
lemma_idx_sorted = [i[0] for i in sorted(enumerate(lemma_len), key=lambda x:x[1]) if i[1] > 80][2:3]

file_lemma = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.validation.lemma.lower')
f_lemma = file_lemma.readlines()
f_token = file_token.readlines()
f_tgt = file_tgt.readlines()

new_lemma = []
new_token = []
new_tgt = []
for idx, sent in enumerate(f_lemma):
    if idx in lemma_idx_sorted:
        new_lemma.append(sent)
        new_token.append(f_token[idx])
        new_tgt.append(f_tgt[idx])

file_new_lemma = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.debug.lemma.lower', 'w')
file_new_lemma.writelines(new_lemma)
file_new_lemma.close()
file_new_token = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.debug.token.lower', 'w')
file_new_token.writelines(new_token)
file_new_token.close()
file_new_tgt = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/tgt.txt.debug.lower', 'w')
file_new_tgt.writelines(new_tgt)
file_new_tgt.close()