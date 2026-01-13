from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.neural_network import MLPClassifier

from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

def train_models(models, X_train, X_test, y_train, y_test):
    best_acc = 0.0
    best_f1 = 0.0
    best_model = None

    for model in models:
        name = model.__class__.__name__
        print(f"===== {name} =====")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")

        print(f"Accuracy   : {acc:.4f}")
        print(f"Macro F1   : {f1:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification report:")
        print(classification_report(y_test, y_pred))
        print("-" * 60)

        if f1 > best_f1:
            best_f1 = f1
            best_acc = acc
            best_model = model

    print(f"\nBest model: {best_model.__class__.__name__} "
          f"(Accuracy = {best_acc:.4f}, Macro F1 = {best_f1:.4f})")

    return best_model



def prepare_train_test(df, text_col="CommentText"):
    numeric_features = ["Likes","Replies","Comment_Length","Month","DayOfWeek","Hour","IsWeekend"]
    y = df["Sentiment"]
    df_train, df_test, y_train, y_test = train_test_split(df,y,test_size=0.2,stratify=y,random_state=42)

    tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True
)
    X_train_text = tfidf.fit_transform(df_train[text_col])
    X_test_text  = tfidf.transform(df_test[text_col])

    X_train_other = df_train[numeric_features].astype(float)
    X_test_other  = df_test[numeric_features ].astype(float)

    X_train = hstack([X_train_text, X_train_other.values])
    X_test  = hstack([X_test_text,  X_test_other.values])

    return X_train, X_test, y_train, y_test, tfidf

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
def prepare_train_test_delete_stop_words(df, text_col="CommentText"):
    custom_stopwords = list(set(ENGLISH_STOP_WORDS) - {"not", "no", "nor"})
    numeric_features = ["Likes","Replies","Comment_Length","Month","DayOfWeek","Hour","IsWeekend"]
    y = df["Sentiment"]
    df_train, df_test, y_train, y_test = train_test_split(df,y,test_size=0.2,stratify=y,random_state=42)

    tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=3,
    sublinear_tf=True,
    stop_words=custom_stopwords
)
    X_train_text = tfidf.fit_transform(df_train[text_col])
    X_test_text  = tfidf.transform(df_test[text_col])

    X_train_other = df_train[numeric_features].astype(float)
    X_test_other  = df_test[numeric_features ].astype(float)

    X_train = hstack([X_train_text, X_train_other.values])
    X_test  = hstack([X_test_text,  X_test_other.values])

    return X_train, X_test, y_train, y_test, tfidf



def prepare_train_test_v2(df, text_col="CommentText"):
    numeric_features = ["Likes","Replies","Comment_Length","Month","DayOfWeek","Hour","IsWeekend"]
    y = df["Sentiment"]
    df_train, df_test, y_train, y_test = train_test_split(df,y,test_size=0.2,stratify=y,random_state=42)

    tfidf = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3,5),
    min_df=3,
    sublinear_tf=True)

    X_train_text = tfidf.fit_transform(df_train[text_col])
    X_test_text  = tfidf.transform(df_test[text_col])

    X_train_other = df_train[numeric_features].astype(float)
    X_test_other  = df_test[numeric_features ].astype(float)

    X_train = hstack([X_train_text, X_train_other.values])
    X_test  = hstack([X_test_text,  X_test_other.values])

    return X_train, X_test, y_train, y_test, tfidf

def prepare_train_test_v3(df, text_col="CommentText"):
    numeric_features = ["Likes","Replies","Comment_Length","Month","DayOfWeek","Hour","IsWeekend"]
    y = df["Sentiment"]
    df_train, df_test, y_train, y_test = train_test_split(df,y,test_size=0.2,stratify=y,random_state=42)

    tfidf = TfidfVectorizer(
    analyzer="char_wb",
    max_features=50000,
    ngram_range=(3,5),
    min_df=3,
    sublinear_tf=True)

    X_train_text = tfidf.fit_transform(df_train[text_col])
    X_test_text  = tfidf.transform(df_test[text_col])

    X_train_other = df_train[numeric_features].astype(float)
    X_test_other  = df_test[numeric_features ].astype(float)

    X_train = hstack([X_train_text, X_train_other.values])
    X_test  = hstack([X_test_text,  X_test_other.values])

    return X_train, X_test, y_train, y_test, tfidf

def prepare_train_test_v3_for_train(df, text_col="CommentText"):
    numeric_features = ["Likes","Comment_Length","Month","DayOfWeek","Hour","IsWeekend"]
    y = df["Sentiment"]
    df_train, df_test, y_train, y_test = train_test_split(df,y,test_size=0.2,stratify=y,random_state=42)

    tfidf = TfidfVectorizer(
    analyzer="char_wb",
    max_features=50000,
    ngram_range=(3,5),
    min_df=3,
    sublinear_tf=True)

    X_train_text = tfidf.fit_transform(df_train[text_col])
    X_test_text  = tfidf.transform(df_test[text_col])

    X_train_other = df_train[numeric_features].astype(float)
    X_test_other  = df_test[numeric_features ].astype(float)

    X_train = hstack([X_train_text, X_train_other.values])
    X_test  = hstack([X_test_text,  X_test_other.values])

    return X_train, X_test, y_train, y_test, tfidf

def prepare_train_test_only_text(df, text_col="CommentText"):
    y = df["Sentiment"]
    df_train, df_test, y_train, y_test = train_test_split(df,y,test_size=0.2,stratify=y,random_state=42)

    tfidf = TfidfVectorizer(
    analyzer="char_wb",
    max_features=50000,
    ngram_range=(3,5),
    min_df=3,
    sublinear_tf=True)

    X_train = tfidf.fit_transform(df_train[text_col])
    X_test  = tfidf.transform(df_test[text_col])

    return X_train, X_test, y_train, y_test, tfidf


import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

def prepare_text_datasets(df, text_col="CommentText"):
    label_col="Sentiment"
    test_size=0.2
    random_state=42
    max_tokens=40000
    seq_len=150
    batch_size=256
    texts = df[text_col].fillna("").astype(str).values
    labels = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=test_size, stratify=labels, random_state=random_state)

    vectorizer = layers.TextVectorization(max_tokens=max_tokens, output_mode="int", output_sequence_length=seq_len)
    vectorizer.adapt(X_train)
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds, vectorizer

def prepare_train_test_two_channel(df, text_col="CommentText"):
    numeric_features = ["Likes","Replies","Comment_Length","Month","DayOfWeek","Hour","IsWeekend"]
    y = df["Sentiment"]
    df_train, df_test, y_train, y_test = train_test_split(df,y,test_size=0.2,stratify=y,random_state=42)

    tfidf_word = TfidfVectorizer(
        analyzer="word",
        max_features=30000,
        ngram_range=(1,2),
        min_df=3,
        sublinear_tf=True,
        stop_words="english"
    )

    tfidf_char = TfidfVectorizer(
        analyzer="char_wb",
        max_features=30000,
        ngram_range=(3,5),
        min_df=3,
        sublinear_tf=True
    )

    X_train_word = tfidf_word.fit_transform(df_train[text_col].fillna("").astype(str))
    X_test_word  = tfidf_word.transform(df_test[text_col].fillna("").astype(str))

    X_train_char = tfidf_char.fit_transform(df_train[text_col].fillna("").astype(str))
    X_test_char  = tfidf_char.transform(df_test[text_col].fillna("").astype(str))

    X_train_text = hstack([X_train_word, X_train_char])
    X_test_text  = hstack([X_test_word,  X_test_char])


    X_train_other = df_train[numeric_features].astype(float)
    X_test_other  = df_test[numeric_features ].astype(float)

    X_train = hstack([X_train_text, X_train_other.values])
    X_test  = hstack([X_test_text,  X_test_other.values])

    return X_train, X_test, y_train, y_test, (tfidf_word, tfidf_char)