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

