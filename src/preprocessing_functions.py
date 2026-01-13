import re
import pandas as pd

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return pd.NA
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def cleaning_text_lenghth(df):
    df = df[(df["Comment_Length"] >= 3) & (df["Comment_Length"] <= 1000)].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def add_comment_length(df):
    df['Comment_Length'] = df['CommentText'].apply(len)
    return df

def encoding_date(df):
    df['PublishedAt'] = pd.to_datetime(df['PublishedAt'])
    df['Month'] = df['PublishedAt'].dt.month
    df['DayOfWeek'] = df['PublishedAt'].dt.dayofweek
    df['Hour'] = df['PublishedAt'].dt.hour
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df.drop(columns=['PublishedAt'], inplace=True)
    return df

import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # mai rapid

NEGATIONS = {"not", "no", "never", "n't"}

def lemmatize_spacy_series(texts):
    out = []
    for doc in nlp.pipe(texts, batch_size=1000):
        tokens = []
        for t in doc:
            if t.lower_ in NEGATIONS:
                tokens.append(t.lower_)
                continue
            if t.is_space:
                continue
            if t.is_punct:
                if t.text in {"!", "?"}:
                    tokens.append(t.text)
                continue
            if t.is_stop:
                continue
            lemma = t.lemma_.lower()
            if lemma == "-pron-":
                lemma = t.lower_
            tokens.append(lemma)
        out.append(" ".join(tokens))
    return out

