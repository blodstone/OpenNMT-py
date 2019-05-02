import torch
from onmt.inputters.datareader_base import DataReaderBase
from torchtext.data import Field, RawField


class TopicDataReader(DataReaderBase):
    def read(self, topics, side, _dir=None):
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        for i, topic in enumerate(topics):
            yield {side: topic['topic'], "indices": i}


def topic_to_tensor(data, vocab):
    return torch.unsqueeze(torch.FloatTensor(data), 0)


def topic_fields(**kwargs):
    topic = Field(use_vocab=False, dtype=torch.float,
        postprocessing=topic_to_tensor, sequential=False)
    return topic


class LemmaField(RawField):
    pass

def lemma_to_topic(data, vocab):
    print()
    pass


def lemma_fields(**kwargs):
    lemma = Field(use_vocab=False, dtype=torch.float,
        postprocessing=lemma_to_topic, sequential=True)
    return lemma