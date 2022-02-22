feature = []
with open("data/feature.csv", "r", encoding='utf-8') as fin:
    for line in fin:
        feature.append(line.replace("\n", "").split(" "))

import pandas as pd
train_df = pd.read_csv("data/train_set.csv", sep='\t', nrows=100)

def caculate(word_count, feature):
    ans = 0
    res = '0'
    for cls in feature:
        tmp = 0
        # print(cls)
        for k in cls[1:]:
            # print(k, word_count[k])
            tmp += word_count[k]
        if tmp > ans:
            res = cls[0]
            ans = tmp
    return res

class Eva():
    def __init__(self):
        self.sum = 0
        self.count = 0

    def evaluate(self, label, pre):
        if label == int(pre):
            self.sum += 1
        self.count += 1
        return self.sum/self.count

from collections import Counter

from sklearn.metrics import f1_score
y_true = []
y_pred = []


eval = Eva()
for line,label in zip(train_df["text"], train_df["label"]):
    # print(line)
    word_count = Counter(line.split(" "))
    # print(word_count)
    pre = caculate(word_count, feature)
    # print("label:", label)
    # print("pre", pre)
    acc = eval.evaluate(label, pre)
    y_true.append(label)
    y_pred.append(int(pre))

print("acc", acc)
f1 = f1_score(y_true, y_pred, average='macro')
print("f1", f1)





def caculate_test():
    test_df = pd.read_csv("data/test_a.csv", sep='\t')

    with open("data/test_a_submit.csv", 'w', encoding='utf-8') as fout:
        fout.writelines("label\n")
        for line in test_df["text"]:
            word_count = Counter(line.split(" "))
            pre = caculate(word_count, feature)
            # print("pre", pre)
            fout.writelines(pre+"\n")
    print("write done!")
caculate_test()
