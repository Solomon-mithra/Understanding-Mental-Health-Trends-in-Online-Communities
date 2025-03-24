import streamlit as st
import pandas as pd
import altair as alt
import requests
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

# -------------------------------
# Configuration for Groq API
# -------------------------------
# Update this URL with the correct endpoint from Groq's documentation.
GROQ_SUMMARIZATION_URL = "https://api.groq.com/v1/summarization"  
GROQ_API_KEY = st.secrets["GROQ"]["API_KEY"]  # Ensure your secrets.toml has this key
model = "llama3-8b-8192"  # Define the model here; adjust as needed

# -------------------------------
# Groq API Summarization Functions
# -------------------------------
def summarize_with_groq(text):
    """
    Calls the Groq API for summarization with parameters set to produce roughly 100 words.
    Expects the API to return a JSON with a key "summary".
    """
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "max_length": 100,  # Set max_length to 100 (approx. 100 words)
        "min_length": 100   # Set min_length to 100 to aim for exactly 100 words
    }
    response = requests.post(GROQ_SUMMARIZATION_URL, headers=headers, json=payload)
    if response.status_code == 200:
        data = response.json()
        return data.get("summary", "")
    else:
        return f"Error: {response.status_code} - {response.text}"

def summarize_with_groq_prompt(sample_text, graph_data, groq_chat):
    """
    Constructs a custom prompt that includes the graph summary data and sample posts,
    then calls the Groq API (via LangChain's ChatGroq model) to generate a concise summary.
    """
    prompt_text = (
        f"Graph Data: {graph_data}\n\n"
        f"Sample Posts: {sample_text}\n\n"
        "Based on the above information, provide a concise summary (about 100 words) that explains the relationship "
        "between the labels shown in the graph and how the sample posts support this pattern."
    )
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="You are a summarization assistant that specializes in analyzing relationship data."),
        HumanMessagePromptTemplate.from_template("{input_text}")
    ])
    
    conversation = prompt | groq_chat
    response = conversation.invoke({"input_text": prompt_text})
    return response.content

# -------------------------------
# Data Loading
# -------------------------------
def load_data():
    file_path = "merged_results.csv"  # Ensure this CSV is in the Streamlit app directory
    df = pd.read_csv(file_path)
    return df

df = load_data()

# -------------------------------
# App Title & Description
# -------------------------------
st.title("Mental Health Trends in Online Communities")
st.write("Analyzing sentiment and topics from online discussions on mental health.")

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filter Data")
topics_selected = st.sidebar.multiselect(
    "Select Topics", 
    options=df["predefined_topic"].unique(),
    default=[]
)
emotions_selected = st.sidebar.multiselect(
    "Select Emotion Labels",
    options=df["emotion_label"].unique(),
    default=[]
)
bert_labels_selected = st.sidebar.multiselect(
    "Select BERT Labels",
    options=df["bert_label"].unique(),
    default=[]
)

# -------------------------------
# Apply Filters
# -------------------------------
filtered_df = df.copy()
if topics_selected:
    filtered_df = filtered_df[filtered_df["predefined_topic"].isin(topics_selected)]
if emotions_selected:
    filtered_df = filtered_df[filtered_df["emotion_label"].isin(emotions_selected)]
if bert_labels_selected:
    filtered_df = filtered_df[filtered_df["bert_label"].isin(bert_labels_selected)]

st.write(f"Showing **{len(filtered_df)}** results after applying filters.")

# -------------------------------
# Display Filtered Data
# -------------------------------
st.subheader("Filtered Data")
columns_to_show = [
    "subreddit",
    "title",
    "body",
    "date_time",
    "bert_label",
    "emotion_label",
    "predefined_topic"
]
st.dataframe(filtered_df[columns_to_show])
st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# Distribution Charts
# -------------------------------
st.subheader("Distribution of Posts by Emotion Label")
emo_counts = filtered_df["emotion_label"].value_counts()
st.bar_chart(emo_counts)
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Distribution of Posts by BERT Label")
bert_counts = filtered_df["bert_label"].value_counts()
st.bar_chart(bert_counts)
st.markdown("<br>", unsafe_allow_html=True)

st.subheader("Distribution of Posts by Predefined Topic")
topic_counts = filtered_df["predefined_topic"].value_counts()
st.bar_chart(topic_counts)
st.markdown("<br>", unsafe_allow_html=True)

# -------------------------------
# Helper: Create Altair Heatmap
# -------------------------------
def create_heatmap(data, x_col, y_col, color_col, title):
    chart = (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(f"{x_col}:O", title=x_col, axis=alt.Axis(labelAngle=-45, labelFontSize=12)),
            y=alt.Y(f"{y_col}:O", title=y_col, axis=alt.Axis(labelFontSize=12)),
            color=alt.Color(f"{color_col}:Q", scale=alt.Scale(scheme="viridis")),
            tooltip=[x_col, y_col, color_col]
        )
        .properties(title=title, width=500, height=400)
    )
    return chart

# -------------------------------
# Initialize Groq Chat Model via LangChain
# -------------------------------
groq_chat = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model)

# -------------------------------
# Relationship Visualizations with Groq Summarization
# -------------------------------

## 1. Emotion Label vs. BERT Label
st.subheader("Relationship: Emotion Label vs. BERT Label")
if not filtered_df.empty:
    ct_emo_bert = pd.crosstab(filtered_df["emotion_label"], filtered_df["bert_label"]).reset_index()
    ct_emo_bert_melted = ct_emo_bert.melt(id_vars="emotion_label", var_name="bert_label", value_name="count")
    chart = create_heatmap(ct_emo_bert_melted, "bert_label", "emotion_label", "count", "Emotion vs. BERT Label")
    st.altair_chart(chart, use_container_width=True)
    
    ct_orig = pd.crosstab(filtered_df["emotion_label"], filtered_df["bert_label"])
    max_idx = ct_orig.stack().idxmax()  # (emotion, bert)
    max_value = ct_orig.stack().max()
    graph_data = f"Posts with emotion label '{max_idx[0]}' and BERT label '{max_idx[1]}' appear {max_value} times."
    st.markdown("**Graph Summary:** " + graph_data)
    
    sample_posts = filtered_df[
        (filtered_df["emotion_label"] == max_idx[0]) & (filtered_df["bert_label"] == max_idx[1])
    ].head(5)
    sample_text = " ".join((sample_posts["title"].fillna("") + ". " + sample_posts["body"].fillna("")).tolist())
    
    try:
        combined_summary = summarize_with_groq_prompt(sample_text, graph_data, groq_chat)
        st.markdown("**Combined Summary:** " + combined_summary)
    except Exception as e:
        st.markdown("**Combined Summary:** Error: " + str(e))
    
    st.markdown("**Sample Posts:**")
    st.write(sample_posts[columns_to_show])
else:
    st.write("No data available for this visualization.")
st.markdown("<br>", unsafe_allow_html=True)

## 2. Emotion Label vs. Predefined Topic
st.subheader("Relationship: Emotion Label vs. Predefined Topic")
if not filtered_df.empty:
    ct_emo_topic = pd.crosstab(filtered_df["emotion_label"], filtered_df["predefined_topic"]).reset_index()
    ct_emo_topic_melted = ct_emo_topic.melt(id_vars="emotion_label", var_name="predefined_topic", value_name="count")
    chart = create_heatmap(ct_emo_topic_melted, "predefined_topic", "emotion_label", "count", "Emotion vs. Predefined Topic")
    st.altair_chart(chart, use_container_width=True)
    
    ct_orig = pd.crosstab(filtered_df["emotion_label"], filtered_df["predefined_topic"])
    max_idx = ct_orig.stack().idxmax()  # (emotion, topic)
    max_value = ct_orig.stack().max()
    graph_data = f"Posts with emotion label '{max_idx[0]}' and predefined topic '{max_idx[1]}' appear {max_value} times."
    st.markdown("**Graph Summary:** " + graph_data)
    
    sample_posts = filtered_df[
        (filtered_df["emotion_label"] == max_idx[0]) & (filtered_df["predefined_topic"] == max_idx[1])
    ].head(5)
    sample_text = " ".join((sample_posts["title"].fillna("") + ". " + sample_posts["body"].fillna("")).tolist())
    
    try:
        combined_summary = summarize_with_groq_prompt(sample_text, graph_data, groq_chat)
        st.markdown("**Combined Summary:** " + combined_summary)
    except Exception as e:
        st.markdown("**Combined Summary:** Error: " + str(e))
    
    st.markdown("**Sample Posts:**")
    st.write(sample_posts[columns_to_show])
else:
    st.write("No data available for this visualization.")
st.markdown("<br>", unsafe_allow_html=True)

## 3. BERT Label vs. Predefined Topic
st.subheader("Relationship: BERT Label vs. Predefined Topic")
if not filtered_df.empty:
    ct_bert_topic = pd.crosstab(filtered_df["bert_label"], filtered_df["predefined_topic"]).reset_index()
    ct_bert_topic_melted = ct_bert_topic.melt(id_vars="bert_label", var_name="predefined_topic", value_name="count")
    chart = create_heatmap(ct_bert_topic_melted, "predefined_topic", "bert_label", "count", "BERT vs. Predefined Topic")
    st.altair_chart(chart, use_container_width=True)
    
    ct_orig = pd.crosstab(filtered_df["bert_label"], filtered_df["predefined_topic"])
    max_idx = ct_orig.stack().idxmax()  # (bert, topic)
    max_value = ct_orig.stack().max()
    graph_data = f"Posts with BERT label '{max_idx[0]}' and predefined topic '{max_idx[1]}' appear {max_value} times."
    st.markdown("**Graph Summary:** " + graph_data)
    
    sample_posts = filtered_df[
        (filtered_df["bert_label"] == max_idx[0]) & (filtered_df["predefined_topic"] == max_idx[1])
    ].head(5)
    sample_text = " ".join((sample_posts["title"].fillna("") + ". " + sample_posts["body"].fillna("")).tolist())
    
    try:
        combined_summary = summarize_with_groq_prompt(sample_text, graph_data, groq_chat)
        st.markdown("**Combined Summary:** " + combined_summary)
    except Exception as e:
        st.markdown("**Combined Summary:** Error: " + str(e))
    
    st.markdown("**Sample Posts:**")
    st.write(sample_posts[columns_to_show])
else:
    st.write("No data available for this visualization.")
