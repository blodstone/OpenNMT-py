import torch
from onmt.inputters.datareader_base import DataReaderBase
from torchtext.data import Field, RawField


class DocTopicDataReader(DataReaderBase):
    def read(self, topics, side, _dir=None):
        assert _dir is None or _dir == "", \
            "Cannot use _dir with TextDataReader."
        for i, topic in enumerate(topics):
            yield {side: topic['topic'], "indices": i}


class DocTopicField(RawField):

    def __init__(self):
        super(DocTopicField, self).__init__()
        self.sequential = False
        self.use_vocab = False

    def process(self, batch, device=None):
        return torch.unsqueeze(
            torch.tensor(batch, dtype=torch.float64, device=device), 1)


class WordTopicField(Field):

    def __init__(self):
        super(WordTopicField, self).__init__()
