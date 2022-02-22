# 这个是[阿里云新闻分类大赛](https://tianchi.aliyun.com/competition/entrance/531810/introduction) 的参赛记录
注意：这个分类比赛的特点是数据是脱敏的，如果想要使用预训练模型的话需要重新训练，这里可以参考另外一个仓库[train_bert_from_egg_with_tensorflow](https://github.com/JackKuo666/train_bert_from_egg_with_tensorflow) 来对脱敏数据进行预训练。

# 依赖
pip install -r requirements.txt

# 0.数据分析
0_data_analysis.py

# 1.使用统计频数做feature，进行baseline的预测
1_test_feature.py
使用统计特征来进行预测：
f1:0.43
# 2.使用机器学习方法
2_machine_learning_test.py

```buildoutcfg
count_vec + ridgeClassifier   f1:0.74
TF-IDF    + ridgeClassifier   f1:0.87
TF-IDF    + SVM               f1:0.88
TF-IDF    + LR                f1:0.83
TF-IDF    + lightgbm          f1:0.89
```

# 3.使用深度学习方法
3_deep_learning_fasttext.py

f1:0.9089[调参后]
```buildoutcfg
    训练10k,测试5k:
            默认参数：               F1:0.825
            lr:5                    F1:0.836
            wordNgrams:4            F1:0.820
            epoch:35                F1:0.821
            loss:ns                 F1:0.8717
            loss:softmax            F1:0.8769
     训练45k,测试5k:                 F1:0.8862
     训练45k,测试5k, lr:5, loss:softmax:  F1:0.9089

```
# 4.使用word2vec 训练embedding
4_gensim_word2vec_word_embedding.py

# 5.使用bert
5_bert.py

代码解释：[阿里天池 NLP 入门赛 TextCNN 方案代码详细注释和流程讲解](https://blog.zhangxiann.com/202008111240/?spm=5176.21852664.0.0.3bf33dd7qs1u3O)
f1: 0.9418

```buildoutcfg
2021-07-20 12:11:00,732 INFO: | epoch   1 | score (90.91, 89.14, 89.99) | f1 89.99 | loss 0.2542 | time 3210.83
2021-07-20 12:11:01,111 INFO: 
              precision    recall  f1-score   support

          科技     0.9216    0.9322    0.9269     35027
          股票     0.9258    0.9370    0.9314     33251
          体育     0.9790    0.9814    0.9802     28283
          娱乐     0.9389    0.9524    0.9456     19920
          时政     0.8832    0.8959    0.8895     13515
          社会     0.8698    0.8619    0.8659     11009
          教育     0.9304    0.9232    0.9268      8987
          财经     0.8718    0.8076    0.8385      7957
          家居     0.8901    0.8890    0.8896      7063
          游戏     0.9050    0.8771    0.8909      5291
          房产     0.9217    0.8911    0.9062      4428
          时尚     0.8879    0.8517    0.8694      2818
          彩票     0.9231    0.8598    0.8903      1633
          星座     0.8793    0.8191    0.8481       818

    accuracy                         0.9233    180000
   macro avg     0.9091    0.8914    0.8999    180000
weighted avg     0.9231    0.9233    0.9231    180000

```

# 6.自己训练bert之后再使用上面的bert模型
f1: 0.895

