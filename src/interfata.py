import os
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import numpy as np
import streamlit as st
#streamlit run src/interfata.py
from API_function import getcomments  # funcția ta
# sau: from api_module import getcomments

# Încarcă modelul+tfidf (când îl ai final)
bundle = joblib.load("Models/sentiment_bundle_lgbm.joblib")
model = bundle["model"]
tfidf = bundle["tfidf"]

st.set_page_config(page_title="YouTube Sentiment", layout="centered")
st.title("YouTube Comments Sentiment Analyzer")

video_id = st.text_input("YouTube videoId (ex: dQw4w9WgXcQ)")


if st.button("Analyze"):
    if not video_id:
        st.error("Please enter a valid videoId.")
        st.stop()

    with st.spinner("Downloading comments..."):
        df = getcomments(video_id)  # ideal: getcomments(video_id, max_comments=max_comments)

    st.write(f"Total comments fetched: {len(df)}")
    if len(df) == 0:
        st.warning("Could not fetch comments or no comments available.")
        st.stop()

    # Când ai modelul final:
    numeric_features = ["Likes","Comment_Length","Month","DayOfWeek","Hour","IsWeekend"]
    

    X_text = tfidf.transform(df["CommentText"])
    
    X_other = csr_matrix(df[numeric_features].astype(np.float32).values)
    X = hstack([X_text, X_other]).tocsr()
    preds = model.predict(X)

    

    

    counts = {
        "Negative": int((preds == 0).sum()),
        "Neutral": int((preds == 1).sum()),
        "Positive": int((preds == 2).sum()),
    }
    total = len(preds)
    percent = {k: v / total * 100 for k, v in counts.items()}

    st.subheader("Results")
    st.json(counts)

    st.subheader("Percentages")
    st.write({k: f"{v:.2f}%" for k, v in percent.items()})

    st.bar_chart({k: v / total for k, v in counts.items()})

    df["pred"] = preds
    label_map = {0:"Negative", 1:"Neutral", 2:"Positive"}

    for cls, name in label_map.items():
        st.subheader(f"Examples - {name}")
        sample = df.loc[df["pred"] == cls, "CommentText"].head(3).tolist()
        for s in sample:
            st.write("-", s)
