import pandas as pd
from sklearn.metrics import f1_score

def fasttext():
    print("1.fasttext")

    # 转换为FastText需要的格式
    # idx = 10000
    # train_df = pd.read_csv('data/train_set.csv', sep='\t', nrows=15000)

    idx = 45000
    train_df = pd.read_csv('data/train_set.csv', sep='\t')

    train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
    train_df[['text', 'label_ft']].iloc[:idx].to_csv('data/fasttext_train.csv', index=None, header=None, sep='\t')

    import fasttext
    model = fasttext.train_supervised('data/fasttext_train.csv', lr=5.0, wordNgrams=2,
                                      verbose=2, minCount=1, epoch=25, loss="softmax")
    # loss: {ns:负采样， hs:霍夫曼softmax, softmax:softmax}
    # 注：这里的softmax 其实是指 softmax激活函数 + 交叉熵损失函数

    val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[idx:]['text']]
    print(f1_score(train_df['label'].values[idx:].astype(str), val_pred, average='macro'))


def kfold():
    print("2.fasttext:使用交叉验证")
    import numpy as np
    import fasttext

    from sklearn.model_selection import KFold
    train_df = pd.read_csv('data/train_set.csv', sep='\t')
    train_df['label_ft'] = '__label__' + train_df['label'].astype(str)

    skf = KFold(n_splits=5, shuffle=True)
    score_max = 0
    for train_index, eval_index in skf.split(train_df['text'], train_df['label_ft']):
        train_df[['text', 'label_ft']].iloc[train_index].to_csv('data/fasttext_train.csv', index=None, sep='\t')
        model = fasttext.train_supervised('data/fasttext_train.csv', lr=1.0, wordNgrams=2, verbose=2, minCount=1, epoch=25, loss='hs')
        val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[eval_index]['text']]
        score = f1_score(train_df['label'].values[eval_index].astype(str), val_pred, average='macro')
        print(score)
        if score > score_max:
            model.save_model("data/best_fasttext_model.pkl")
            score_max = score
    print("best score is:"+score_max)


# kfold()
def gen_result():
    import fasttext
    import pandas as pd
    model = fasttext.load_model("data/best_fasttext_model.pkl")
    test_df = pd.read_csv("data/test_a.csv", sep='\t')

    with open("data/test_c_submit.csv", 'w', encoding='utf-8') as fout:
        fout.writelines("label\n")
        for line in test_df["text"]:
            pre = model.predict(line)[0][0].split('__')[-1]
            fout.writelines(pre+"\n")
    print("write done!")

gen_result()
