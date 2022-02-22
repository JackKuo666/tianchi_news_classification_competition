import logging
import random
import pandas as pd
import numpy as np
import torch
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# #############################################
# # split data to 10 fold
# fold_num = 10
# data_file = '/home/featurize/data/train_set.csv'


# def all_data2fold(fold_num, num=10000):
#     fold_data = []
#     f = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
#     texts = f['text'].tolist()[:num]
#     labels = f['label'].tolist()[:num]

#     total = len(labels)

#     index = list(range(total))
#     np.random.shuffle(index)

#     all_texts = []
#     all_labels = []
#     for i in index:
#         all_texts.append(texts[i])
#         all_labels.append(labels[i])

#     label2id = {}
#     for i in range(total):
#         label = str(all_labels[i])
#         if label not in label2id:
#             label2id[label] = [i]
#         else:
#             label2id[label].append(i)

#     all_index = [[] for _ in range(fold_num)]
#     for label, data in label2id.items():
#         # print(label, len(data))
#         batch_size = int(len(data) / fold_num)
#         other = len(data) - batch_size * fold_num
#         for i in range(fold_num):
#             cur_batch_size = batch_size + 1 if i < other else batch_size
#             # print(cur_batch_size)
#             batch_data = [data[i * batch_size + b] for b in range(cur_batch_size)]
#             all_index[i].extend(batch_data)

#     batch_size = int(total / fold_num)
#     other_texts = []
#     other_labels = []
#     other_num = 0
#     start = 0
#     for fold in range(fold_num):
#         num = len(all_index[fold])
#         texts = [all_texts[i] for i in all_index[fold]]
#         labels = [all_labels[i] for i in all_index[fold]]

#         if num > batch_size:
#             fold_texts = texts[:batch_size]
#             other_texts.extend(texts[batch_size:])
#             fold_labels = labels[:batch_size]
#             other_labels.extend(labels[batch_size:])
#             other_num += num - batch_size
#         elif num < batch_size:
#             end = start + batch_size - num
#             fold_texts = texts + other_texts[start: end]
#             fold_labels = labels + other_labels[start: end]
#             start = end
#         else:
#             fold_texts = texts
#             fold_labels = labels

#         assert batch_size == len(fold_labels)

#         # shuffle
#         index = list(range(batch_size))
#         np.random.shuffle(index)

#         shuffle_fold_texts = []
#         shuffle_fold_labels = []
#         for i in index:
#             shuffle_fold_texts.append(fold_texts[i])
#             shuffle_fold_labels.append(fold_labels[i])

#         data = {'label': shuffle_fold_labels, 'text': shuffle_fold_texts}
#         fold_data.append(data)

#     logging.info("Fold lens %s", str([len(data['label']) for data in fold_data]))

#     return fold_data


# fold_data = all_data2fold(10,200000)



# ################################
# # build train, dev, test data
# fold_id = 9

# # dev
# dev_data = fold_data[fold_id]

# # train
# train_texts = []
# train_labels = []
# for i in range(0, fold_id):
#     data = fold_data[i]
#     train_texts.extend(data['text'])
#     train_labels.extend(data['label'])

# train_data = {'label': train_labels, 'text': train_texts}

# # test
# test_data_file = '/home/featurize/data/test_a.csv'
# f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
# texts = f['text'].tolist()
# test_data = {'label': [0] * len(texts), 'text': texts}


# train
train_data_file = '/home/featurize/data/train_dev_data/train_data.csv'
f = pd.read_csv(train_data_file, sep='\t', encoding='UTF-8')
train_data = {'label': f["label"].tolist(), 'text': f["text"].tolist()}

# dev
dev_data_file = '/home/featurize/data/train_dev_data/dev_data.csv'
f = pd.read_csv(dev_data_file, sep='\t', encoding='UTF-8')
dev_data = {'label': f["label"].tolist(), 'text': f["text"].tolist()}

# test
test_data_file = '/home/featurize/data/test_a.csv'
f = pd.read_csv(test_data_file, sep='\t', encoding='UTF-8')
texts = f['text'].tolist()
test_data = {'label': [0] * len(texts), 'text': texts}
