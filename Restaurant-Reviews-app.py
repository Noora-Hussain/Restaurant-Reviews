import pandas as pd
import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Restaurant Review", layout="wide")

st.title("Restaurant Review ")
st.write("Analyze restaurant feedback")


@st.cache_resource
def load_sentiment_model(Model_name):
  return pipeline("text-classification", model=Model_name)


@st.cache_resource
def load_zeroshot_model():
  return pipeline(
      "zero-shot-classification",
      model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
  )


st.sidebar.header("Settings")
sentiment_model_choice = st.sidebar.selectbox(
    "Choose Sentiment Model:",
    [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    ],
)

st.header("Input Review Text")
user_input = st.text_area(
    "Enter a restaurant review:",
    (
        "Amazing ambiance and the local dishes were exceptionally delicious,"
        " excellent service overall!"
    ),
)

candidate_topics = [
    "Food Quality",
    "Service",
    "Price",
    "Cleanliness",
    "Atmosphere",
    "Location",
    "Waiting Time",
]

if st.button("Analyze"):
  if user_input.strip() != "":
    Sentiment_Pipe = load_sentiment_model(sentiment_model_choice)
    Zeroshot_Pipe = load_zeroshot_model()

    sent_res = Sentiment_Pipe(user_input)[0]
    topic_res = Zeroshot_Pipe(user_input, candidate_labels=candidate_topics)

    col1, col2 = st.columns(2)

    with col1:
      st.subheader("Task A: Sentiment Result")
      st.metric("Predicted Sentiment", sent_res["label"].capitalize())
      st.metric("Confidence Score", f"{sent_res['score']:.4f}")
      st.json({
          "detected_sentiment": sent_res["label"],
          "confidence_value": float(sent_res["score"]),
          "processing_engine": "transformer_base_model",
      })

    with col2:
      st.subheader("Task B: Aspect Categorization")
      primary_aspect = topic_res["labels"][0].replace("_", " ").title()
      aspect_probability = topic_res["scores"][0]

      st.metric("Classified Aspect", primary_aspect)
      st.metric("Aspect Confidence", f"{aspect_probability:.4f}")

      st.json({
          "primary_category": primary_aspect,
          "confidence_score": float(aspect_probability),
          "model_architecture": "zero_shot_transformer",
      })

      results_table = pd.DataFrame({
          "Aspect / Topic": topic_res["labels"],
          "Probability Score": topic_res["scores"],
      })

      st.markdown("### All Topics Probability Breakdown")
      st.dataframe(results_table, use_container_width=True)

  else:
    st.error("⚠️ Please type a valid restaurant review before clicking analyze.")
