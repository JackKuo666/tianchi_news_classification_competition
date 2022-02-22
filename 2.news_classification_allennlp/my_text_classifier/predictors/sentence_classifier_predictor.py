from allennlp.data import DatasetReader, Instance
from allennlp.predictors import Predictor
from overrides import overrides
from allennlp.common.util import JsonDict
from typing import List


@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)

    @overrides
    def load_line(self, line: str) -> JsonDict:
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        return {"sentence": line}

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        # todo:这里也可以在outputs中加入input_text来对比分析数据
        return str(outputs["label"]) + "\n"

    @overrides
    def predict_batch_json(self, inputs: List[JsonDict]) -> List[JsonDict]:
        instances = self._batch_json_to_instances(inputs)
        outputs = self.predict_batch_instance(instances)
        # print("inputs", inputs)
        outputs = [{"label": i["label"]} for i in outputs]
        # print("outputs", outputs)

        return outputs