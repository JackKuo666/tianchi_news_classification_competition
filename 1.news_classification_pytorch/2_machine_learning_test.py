def count_vec_test():
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    vectorizer = CountVectorizer()
    bag_of_words_vec = vectorizer.fit_transform(corpus).toarray()
    print("bag_of_words_vec\n", bag_of_words_vec)


def count_vec():
    print("\n1.Count Vectors + RidgeClassifier")

    import pandas as pd

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import f1_score

    train_df = pd.read_csv('data/train_set.csv', sep='\t', nrows=15000)

    vectorizer = CountVectorizer(max_features=3000)
    train_test = vectorizer.fit_transform(train_df['text'])
    print("train_test[:1]", train_test[0].toarray()[0].size, train_test[0].toarray()[0][:15])

    clf = RidgeClassifier()
    clf.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = clf.predict(train_test[10000:])
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))


def tf_idf():
    print("\n2.TF-IDF +  RidgeClassifier")

    import pandas as pd

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import f1_score

    train_df = pd.read_csv('data/train_set.csv', sep='\t', nrows=15000)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
    train_test = tfidf.fit_transform(train_df['text'])
    print("train_test[:1]", train_test[0].toarray()[0].size, train_test[0].toarray()[0][:15])

    clf = RidgeClassifier()
    clf.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = clf.predict(train_test[10000:])
    print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))

tf_idf()

def svm():
    import pandas as pd

    from sklearn.multiclass import OneVsRestClassifier
    from sklearn import svm

    print("\n3.TF-IDF +  svm")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Lasso
    from sklearn.metrics import f1_score

    train_df = pd.read_csv('data/train_set.csv', sep='\t', nrows=15000)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
    train_test = tfidf.fit_transform(train_df['text'])
    print("train_test[:1]", train_test[0].toarray()[0].size, train_test[0].toarray()[0][:15])

    model = OneVsRestClassifier(svm.LinearSVC(random_state=0, verbose=1))
    model.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = model.predict(train_test[10000:])  # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]

    print("\nf1:", f1_score(train_df['label'].values[10000:], val_pred, average='macro'))





def LR():
    import pandas as pd

    print("\n4.TF-IDF + LR")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    train_df = pd.read_csv('data/train_set.csv', sep='\t', nrows=15000)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
    train_test = tfidf.fit_transform(train_df['text'])
    print("train_test[:1]", train_test[0].toarray()[0].size, train_test[0].toarray()[0][:15])

    model = LogisticRegression(random_state=0, solver='sag', multi_class='ovr', verbose=1)
    model.fit(train_test[:10000], train_df['label'].values[:10000])

    val_pred = model.predict(train_test[10000:])  # 计算属于各个类别的概率，返回值的shape = [n_samples, n_classes]

    print("\nf1:", f1_score(train_df['label'].values[10000:], val_pred, average='macro'))



def lightgbm():
    import pandas as pd

    print("\n5.TF-IDF + lightgbm")

    from sklearn.feature_extraction.text import TfidfVectorizer
    import lightgbm as lgb
    from sklearn.metrics import f1_score

    train_df = pd.read_csv('data/train_set.csv', sep='\t', nrows=15000)

    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=3000)
    train_test = tfidf.fit_transform(train_df['text'])
    print("train_test[:1]", train_test[0].toarray()[0].size, train_test[0].toarray()[0][:15])

    train_data = lgb.Dataset(train_test[:10000], label=train_df['label'].values[:10000])
    validation_data = lgb.Dataset(train_test[10000:], label=train_df['label'].values[10000:])
    params = {
        'learning_rate': 0.1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.2,
        'max_depth': 6,
        'objective': 'multiclass',
        'num_class': 14,
    }
    clf = lgb.train(params, train_data, valid_sets=[validation_data])
    y_pred_pa = clf.predict(train_test[10000:])  # !!!注意lgm预测的是分数，类似 sklearn的predict_proba
    y_pred = y_pred_pa.argmax(axis=1)

    print("\nf1:", f1_score(train_df['label'].values[10000:], y_pred, average='macro'))

