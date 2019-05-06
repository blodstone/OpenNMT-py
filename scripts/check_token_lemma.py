lemma = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.train.lemma', 'r')
token = open('/home/acp16hh/Projects/Research/Experiments/Exp_Freya_Topic_Summ/Topic_Summ/data/bbc-split/src.train.token', 'r')
lemma_l = [len(l.split()) for l in lemma.readlines()]
token_l = [len(l.split()) for l in token.readlines()]
for i in range(len(lemma_l)):
    print("{}: {}".format(i, lemma_l[i]))
    if lemma_l[i] != token_l[i]:
        print("{} {}".format(lemma_l[i], token_l[i]))
