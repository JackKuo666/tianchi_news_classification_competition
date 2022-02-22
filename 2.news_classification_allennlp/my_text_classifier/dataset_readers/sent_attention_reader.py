from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import LabelField, TextField, ListField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer


@DatasetReader.register("sent_attention_reader")
class SentAttentionReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        # 经过分析大部分文章在1000字到2000字之间，我们的预训练的bert是256长度的，这里将文章分为8句，然后分别token,index,然后传给模型进行后续训练
        texts = text.strip().split(" ")

        tmp = []
        for i in range(0, len(texts), self.max_tokens):
            tmp.append(texts[i: i + self.max_tokens])
        if len(tmp) > 8:
            tmp = tmp[:4]+tmp[-4:]
        if len(tmp) < 8:
            tmp += (8 - len(tmp))*[]

        list_tmp = []
        for line in tmp:
            text = " ".join(line)
            tokens = self.tokenizer.tokenize(text)
            if self.max_tokens:
                tokens = tokens[: self.max_tokens]
            text_field = TextField(tokens, self.token_indexers)
            list_tmp.append(text_field)
        fields = {"text": ListField(list_tmp)}

        if label:
            fields["label"] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                sentiment, text = line.strip().split("\t")
                if text == "text" or sentiment == "label":
                    continue
                yield self.text_to_instance(text, sentiment)
