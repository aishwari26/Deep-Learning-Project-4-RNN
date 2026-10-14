import streamlit as st
import torch
import torch.nn as nn
import requests
import pickle
from collections import Counter
from urllib.parse import urlparse, parse_qs


# =========================================================
# 1. MODEL
# =========================================================

class YouTubeGRUModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=64):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim
        )

        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            1
        )

    def forward(self, x):
        x = self.embedding(x)

        output, hidden = self.gru(x)

        last_hidden = hidden[-1]

        return self.fc(last_hidden).squeeze(1)


# =========================================================
# 2. LOAD YOUTUBE VOCABULARY
# =========================================================

with open("youtube_word_to_id.pkl", "rb") as f:
    youtube_word_to_id = pickle.load(f)


# =========================================================
# 3. TOKENIZATION
# =========================================================

def tokenize(text):
    return text.lower().split()


def pad_sequence(encoded, max_len=200):

    if len(encoded) < max_len:

        encoded = encoded + [
            youtube_word_to_id["<PAD>"]
        ] * (max_len - len(encoded))

    else:

        encoded = encoded[:max_len]

    return encoded


# =========================================================
# 4. LOAD TRAINED MODEL
# =========================================================

youtube_model = YouTubeGRUModel(
    vocab_size=len(youtube_word_to_id)
)

youtube_model.load_state_dict(
    torch.load(
        "best_youtube_gru.pth",
        map_location="cpu"
    )
)

youtube_model.eval()


# =========================================================
# 5. SENTIMENT PREDICTION
# =========================================================

def predict_sentiment(comment):

    tokens = tokenize(comment)

    encoded = [
        youtube_word_to_id.get(
            word,
            youtube_word_to_id["<UNK>"]
        )
        for word in tokens
    ]

    padded = pad_sequence(encoded)

    x = torch.tensor(
        padded,
        dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():

        logits = youtube_model(x)

        positive_probability = torch.sigmoid(
            logits
        ).item()

    if positive_probability >= 0.5:

        sentiment = "Positive"
        confidence = positive_probability

    else:

        sentiment = "Negative"
        confidence = 1 - positive_probability

    return sentiment, confidence


# =========================================================
# 6. EXTRACT VIDEO ID FROM URL
# =========================================================

def extract_video_id(url):

    parsed_url = urlparse(url)

    # youtube.com/watch?v=XXXXXXXXXXX
    if parsed_url.hostname in [
        "www.youtube.com",
        "youtube.com",
        "m.youtube.com"
    ]:

        video_id = parse_qs(
            parsed_url.query
        ).get("v")

        if video_id:
            return video_id[0]

    # youtu.be/XXXXXXXXXXX
    if parsed_url.hostname == "youtu.be":

        return parsed_url.path.lstrip("/")

    return None


# =========================================================
# 7. GET YOUTUBE COMMENTS
# =========================================================

def get_youtube_comments(
    video_id,
    api_key,
    max_comments=100
):

    comments = []
    page_token = None

    while len(comments) < max_comments:

        url = (
            "https://www.googleapis.com/"
            "youtube/v3/commentThreads"
        )

        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(
                100,
                max_comments - len(comments)
            ),
            "textFormat": "plainText",
            "key": api_key
        }

        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        for item in data.get("items", []):

            comment = (
                item["snippet"]
                ["topLevelComment"]
                ["snippet"]
                ["textDisplay"]
            )

            comments.append(comment)

        page_token = data.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return comments


# =========================================================
# 8. PREDICT MULTIPLE COMMENTS
# =========================================================

def predict_comments(comments):

    results = []

    for comment in comments:

        sentiment, confidence = predict_sentiment(
            comment
        )

        results.append({
            "Comment": comment,
            "Sentiment": sentiment,
            "Probability": confidence
        })

    return results


# =========================================================
# 9. STREAMLIT PAGE
# =========================================================

st.set_page_config(
    page_title="YouTube Comment Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

st.title("💬 YouTube Comment Sentiment Analyzer")

st.write(
    "Analyze English and Hinglish comments "
    "as Positive or Negative."
)


# =========================================================
# 10. SINGLE COMMENT ANALYSIS
# =========================================================

st.header("Analyze a Single Comment")

comment_input = st.text_area(
    "Enter your comment:",
    placeholder="Example: This video is amazing bhai!"
)

if st.button("Analyze Comment"):

    if comment_input.strip() == "":

        st.warning(
            "Please enter a comment."
        )

    else:

        sentiment, confidence = predict_sentiment(
            comment_input
        )

        st.subheader("Prediction")

        if sentiment == "Positive":

            st.success(
                f"Positive — "
                f"Confidence: {confidence * 100:.2f}%"
            )

        else:

            st.error(
                f"Negative — "
                f"Confidence: {confidence * 100:.2f}%"
            )


# =========================================================
# 11. YOUTUBE VIDEO ANALYSIS
# =========================================================

st.header("Analyze YouTube Video Comments")

api_key = st.text_input(
    "YouTube API Key",
    type="password"
)

youtube_url = st.text_input(
    "YouTube Video URL",
    placeholder="https://www.youtube.com/watch?v=..."
)

max_comments = st.slider(
    "Number of comments to analyze",
    min_value=10,
    max_value=100,
    value=100,
    step=10
)


if st.button("Analyze YouTube Video"):

    if api_key.strip() == "":

        st.warning(
            "Please enter your YouTube API key."
        )

    elif youtube_url.strip() == "":

        st.warning(
            "Please enter a YouTube video URL."
        )

    else:

        video_id = extract_video_id(
            youtube_url
        )

        if video_id is None:

            st.error(
                "Invalid YouTube URL."
            )

        else:

            try:

                with st.spinner(
                    "Fetching YouTube comments..."
                ):

                    comments = get_youtube_comments(
                        video_id,
                        api_key,
                        max_comments
                    )

                if len(comments) == 0:

                    st.warning(
                        "No comments were found for this video."
                    )

                else:

                    results = predict_comments(
                        comments
                    )

                    sentiment_counts = Counter(
                        result["Sentiment"]
                        for result in results
                    )

                    total = len(results)

                    positive = sentiment_counts[
                        "Positive"
                    ]

                    negative = sentiment_counts[
                        "Negative"
                    ]

                    st.success(
                        f"{total} comments analyzed."
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.metric(
                            "Positive",
                            f"{positive} "
                            f"({positive / total * 100:.1f}%)"
                        )

                    with col2:

                        st.metric(
                            "Negative",
                            f"{negative} "
                            f"({negative / total * 100:.1f}%)"
                        )

                    st.subheader(
                        "Comment Predictions"
                    )

                    for i, result in enumerate(
                        results,
                        1
                    ):

                        st.write(
                            f"**{i}. {result['Comment']}**"
                        )

                        if result["Sentiment"] == "Positive":

                            st.success(
                                f"Positive — "
                                f"Confidence: "
                                f"{result['Probability'] * 100:.2f}%"
                            )

                        else:

                            st.error(
                                f"Negative — "
                                f"Confidence: "
                                f"{result['Probability'] * 100:.2f}%"
                            )

            except requests.exceptions.HTTPError as e:

                st.error(
                    f"YouTube API error: {e}"
                )

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )