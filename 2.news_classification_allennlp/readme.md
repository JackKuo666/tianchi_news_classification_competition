# 1.依赖 
python == 3.8.0

allennlp == 2.4.0

pip install allennlp -i https://pypi.tuna.tsinghua.edu.cn/simple
# 1.1.使用lazy
`注意`：在使用大数据进行训练的时候使用lazy模式是极其重要的，但是记得使用lazy模式之前需要进行数据的按照label的数量进行10fold，使得在训练的数据整体上是分布均匀的。

这里需要知道默认的dataset_loader是：multiprocess_data_loader.py
而这个里面需要指定："max_instances_in_memory": 80,才能使用lazy

例子：
```
"data_loader": {
    "batch_size": 8,
    "max_instances_in_memory": 80,
    "cuda_device": 0,
    "shuffle": true
},
```

# 1.2.build vocab
跟lazy同样道理的是 voacab的构建：由于训练数据比较大的时候，在每一次训练时从头构建vocab（需要完整遍历一边所有的数据集）是比较耗时的，所以我们这里手动构建一个vocab,然后在每次修改模型，训练模型的时候直接load就行了。

1、在jsonnet中设置：
```buildoutcfg
"datasets_for_vocab_creation": ["train","validation"],
```
2、提前使用build-vocab命令生成vocab.tar.gz
```buildoutcfg
allennlp build-vocab scripts/my_text_classifier.jsonnet data/vocab.tar.gz --include-package my_text_classifier
```
我们的生成的位置在：`data/vocab.tar.gz`

这里也可以使用自己的脚本生成，然后仿照`vocab.tar.gz`填入就行了,结构如下：
```
-- vocab
|___ labels.txt
|___ non_padded_namespaces.txt
|___ tokens.txt
```

3、只有生成之后才能在jsonnet中设置：
```buildoutcfg
    "vocabulary":{
        "type": "from_files",
        "directory": "data/vocab_model.tar.gz"
    },
```
4、再次训练的时候就是从这里加载vocab了

# 2.训练
##  2.1 embedding+bag_of_embedding

### train
```buildoutcfg
allennlp train scripts/my_text_classifier.jsonnet --serialization-dir checkpoint --include-package my_text_classifier -f
```
gpu的放在了：data/model.tar.gz
### eval
验证cpu训练的结果
```buildoutcfg
allennlp evaluate checkpoint/model.tar.gz data/news_classification/dev_data.csv --include-package my_text_classifier
```
验证gpu训练的结果：0.8679，放在`data/model.tar.gz`
```buildoutcfg
allennlp evaluate data/model.tar.gz data/news_classification/dev_data.csv --include-package my_text_classifier

```
# todo
1、把 predict 改为预测保存为文本的 done!
# 注意：这里需要修改两个地方：一个是自定义model的class需要写一个`make_output_human_readable`
                        一个是自定义的predictor需要修改一下load line
2、上传本次结果 done 0.8679
3、修改不同的输入：要两头的 todo
4、修改模型：输入文章，先有一个txt2sentce,然后sent2vec
5、使用bert模型：这里需要先完成bert预训练，然后再使用

### predict 
cpu 训练结果进行预测
```buildoutcfg
allennlp predict checkpoint/model.tar.gz data/news_classification/test_a.csv --output-file data/news_classification/predict_result.csv --batch-size 8 --include-package my_text_classifier --predictor sentence_classifier --silent
```
gpu 训练结果进行预测
```buildoutcfg
allennlp predict  checkpoint_sent/model.tar.gz data/news_classification/test_a.csv --output-file data/news_classification/predict_result.csv --batch-size 8 --cuda-device 0--include-package my_text_classifier --predictor sentence_classifier --silent
```

## 2.2 使用 allennlp train 训练 bert embedding+bert pool
这里bert-mini 是别人训练好的
### train
```buildoutcfg
allennlp train scripts/my_text_classifier_bert.jsonnet --serialization-dir checkpoint --include-package my_text_classifier -f
```

### predict
```buildoutcfg
allennlp predict checkpoint/model.tar.gz data/news_classification/test_a.csv --include-package my_text_classifier --predictor sentence_classifier
```
gpu:
```buildoutcfg
allennlp predict data/2.bert-mini+bert_trian_2_epoch_acc_0.91085_提交测试acc_/model.tar.gz data/news_classification/test_a.csv --output-file data/news_classification/predict_result.csv --batch-size 8 --include-package my_text_classifier --predictor sentence_classifier --silent
```



## 2.3 使用bert-mini + sent attention
### train
```buildoutcfg
allennlp train scripts/my_text_classifier_sent_attention.jsonnet -s checkpoint --include-package my_text_classifier -f
```
复现了 人家的结果 0.9469 
