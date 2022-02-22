from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
import numpy as np


# build sent encoder
class SentEncoder(torch.nn.Module):
    def __init__(self, sent_rep_size, sent_hidden_size=256, sent_num_layers=2, dropout=0.15):
        super(SentEncoder, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.sent_lstm = torch.nn.LSTM(
            input_size=sent_rep_size,
            hidden_size=sent_hidden_size,
            num_layers=sent_num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, sent_reps, sent_masks):
        # sent_reps:  b x doc_len x sent_rep_size
        # sent_masks: b x doc_len

        sent_hiddens, _ = self.sent_lstm(sent_reps)  # b x doc_len x hidden*2
        sent_hiddens = sent_hiddens * sent_masks.unsqueeze(2)

        if self.training:
            sent_hiddens = self.dropout(sent_hiddens)

        return sent_hiddens


class Attention(torch.nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight.data.normal_(mean=0.0, std=0.05)

        self.bias = torch.nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias.data.copy_(torch.from_numpy(b))

        self.query = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.query.data.normal_(mean=0.0, std=0.05)

    def forward(self, batch_hidden, batch_masks):
        # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
        # batch_masks:  b x len

        # linear
        key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden

        # compute attention
        outputs = torch.matmul(key, self.query)  # b x len    # 注意，由于这里的 q 的维度是【hidden】,所以结果计算的结果是每一个sent 所占句子的比重
        # 所以，这里面没有attention,只是每个句子，只是计算哪个句子占整篇文章的重要性。换句话说是由句向量到文章向量的计算。

        masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))

        attn_scores = torch.nn.functional.softmax(masked_outputs, dim=1)  # b x len

        # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
        masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)

        batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden

        return batch_outputs, attn_scores


@Model.register("sent_attention_classifier")
class SentAttentionClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        sent_hidden_size = 256
        sent_num_layers = 2
        self.doc_rep_size = sent_hidden_size * sent_num_layers

        self.sent_encoder = SentEncoder(sent_rep_size=encoder.get_output_dim(), sent_hidden_size=sent_hidden_size,
                                        sent_num_layers=sent_num_layers, dropout=0.15)
        self.sent_attention = Attention(self.doc_rep_size)

        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(self.doc_rep_size, num_labels)
        self.accuracy = CategoricalAccuracy()
        self.accuracy_2 = FBetaMeasure(average="macro")

    def forward(self, text, label: torch.Tensor = None) -> Dict[str, torch.Tensor]:

        # batch_size x doc_len x sent_len
        batch_size, max_doc_len, max_sent_len = text["bert"]["token_ids"].shape

        # 这里 sen_num = batch_size x doc_len
        # sen_num x sent_len
        text = {"bert": {"token_ids": text["bert"]["token_ids"].view(batch_size * max_doc_len, max_sent_len),
                         "mask": text["bert"]["mask"].view(batch_size * max_doc_len, max_sent_len),
                         "type_ids": text["bert"]["type_ids"].view(batch_size * max_doc_len, max_sent_len)}}

        # Shape: (sen_num, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (sen_num, sent_len)
        mask = util.get_text_field_mask(text)
        # Shape: (sen_num, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # b x doc_len x encoding_dim
        encoded_text = encoded_text.view(batch_size, max_doc_len, encoded_text.shape[-1])
        # b x doc_len
        sent_masks = text["bert"]["mask"].view(batch_size, max_doc_len, max_sent_len).any(2).float()

        # b x doc_len x doc_rep_size
        sent_hiddens = self.sent_encoder(encoded_text, sent_masks)
        # b x doc_rep_size
        doc_reps, atten_scores = self.sent_attention(sent_hiddens, sent_masks)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(doc_reps)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            self.accuracy_2(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset),
                "f1": self.accuracy_2.get_metric(reset)["fscore"]}

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        label = torch.argmax(output_dict["probs"], dim=1)
        label = [self.vocab.get_token_from_index(int(i), "labels") for i in label]
        output_dict["label"] = label
        return output_dict