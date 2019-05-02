import torch
from onmt.inputters.datareader_base import DataReaderBase
from torchtext.data import Field


class TopicDataReader(DataReaderBase):
    def read(self, topics, side, _dir=None):
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        for i, topic in enumerate(topics):
            yield {side: topic['topic'], "indices": i}


def batch_img(data, vocab):
    return torch.FloatTensor(data)


def topic_fields(**kwargs):
    topic = Field(use_vocab=False, dtype=torch.float,
        postprocessing=batch_img, sequential=False)
    return topic
