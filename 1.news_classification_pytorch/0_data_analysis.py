import pandas as pd
train_df = pd.read_csv("data/train_set.csv", sep='\t')#, nrows=100)

# print("1.查看训练数据")
# print(train_df.head())
#
#
# train_df["text_len"] = train_df["text"].apply(lambda x: len(x.split(" ")))
# print("2.查看文本长短分布情况")
# print(train_df["text_len"].head())
# print(train_df["text_len"].describe())
#
# import matplotlib.pyplot as plt
# _ = plt.hist(train_df['text_len'], bins=200)
# plt.xlabel('Text char count')
# plt.title("Histogram of char count")
# plt.show()
#
#
# print("3.新闻类别统计")
# train_df["label"].value_counts().plot(kind='bar')
# plt.title("news class count")
# plt.xlabel("category")
# plt.show()
#
print("4.字符统计")
from collections import Counter
all_lines = ' '.join(list(train_df["text"]))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse=True)
print(len(word_count))
print(word_count[:5])
print(word_count[-1])
all_keys = [k[0] for k in word_count[:20]]
print("all_keys", all_keys)
print(word_count[-5:])

# print("统计同一个字符出现在多少个句子")
# from collections import Counter
# train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
# all_lines = ' '.join(list(train_df['text_unique']))
# word_count = Counter(all_lines.split(" "))
# word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)
# print(word_count[:5])
#


# feature = []
# print("作业：统计每类新闻中次数最多的字符")
# from collections import Counter
# for i in range(14):
#     # print(i)
#     all_lines = ' '.join(list(train_df[train_df["label"] == i]["text"]))
#     word_count = Counter(all_lines.split(" "))
#     word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
#     # print(len(word_count))
#     # print(i, word_count[:20])
#     sub = [k for k in word_count[:20] if k[0] not in all_keys]
#     # print("sub", sub)
#     tmp = [k[0] for k in sub]
#     feature.append(" ".join([str(i)]+tmp))
# # exit()
# with open("data/feature.csv", "w", encoding='utf-8') as fout:
#     for line in feature:
#         fout.writelines(line+"\n")
#     print("save feature.csv done ")