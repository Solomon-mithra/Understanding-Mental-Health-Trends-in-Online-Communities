import streamlit as st
import pandas as pd
import altair as alt
import requests
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_groq import ChatGroq

# -------------------------------
# Configuration for Groq API
# -------------------------------
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
        "max_length": 100,  # approx. 100 words
        "min_length": 100
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

# -------------------------------
# Timeline (Bar) Graphs: Months vs Labels with Year Selection
# -------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Timeline: Monthly Trends (Months vs Labels)")

# Convert date_time to datetime and extract year and month
df_time = filtered_df.copy()
df_time['date'] = pd.to_datetime(df_time['date_time'], errors='coerce')
df_time = df_time.dropna(subset=['date'])
df_time['year'] = df_time['date'].dt.year
df_time['month'] = df_time['date'].dt.month_name()

# Create a dropdown for selecting the year (default to current year if available)
available_years = sorted(df_time['year'].unique())
default_year = datetime.now().year if datetime.now().year in available_years else available_years[0]
selected_year = st.selectbox("Select Year", options=available_years, index=available_years.index(default_year))

# Filter data for the selected year
df_year = df_time[df_time['year'] == selected_year]
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']

# Timeline Bar Graph for Emotion Label:
st.markdown("**Timeline: Months vs Emotion Label**")
timeline_emotion = df_year.groupby(['month', 'emotion_label']).size().reset_index(name='count')
timeline_emotion['month'] = pd.Categorical(timeline_emotion['month'], categories=months_order, ordered=True)
chart_emotion_timeline = alt.Chart(timeline_emotion).mark_bar().encode(
    x=alt.X('month:N', title='Month', sort=months_order),
    xOffset='emotion_label:N',
    y=alt.Y('count:Q', title='Number of Posts'),
    color='emotion_label:N',
    tooltip=['month', 'emotion_label', 'count']
).properties(width=700, height=300)
st.altair_chart(chart_emotion_timeline, use_container_width=True)

if not timeline_emotion.empty:
    ct_emotion = timeline_emotion.groupby('emotion_label')['count'].sum().reset_index()
    max_emotion = ct_emotion.loc[ct_emotion['count'].idxmax()]
    graph_data_emotion = f"In {selected_year}, posts with emotion label '{max_emotion['emotion_label']}' total {max_emotion['count']}."
    sample_posts_emotion = df_year[df_year["emotion_label"] == max_emotion['emotion_label']].head(5)
    sample_text_emotion = " ".join((sample_posts_emotion["title"].fillna("") + ". " + sample_posts_emotion["body"].fillna("")).tolist())
    try:
        timeline_summary = summarize_with_groq_prompt(sample_text_emotion, graph_data_emotion, groq_chat)
        st.markdown("**Timeline Summary (Emotion):** " + timeline_summary)
    except Exception as e:
        st.markdown("**Timeline Summary (Emotion):** Error: " + str(e))
st.markdown("<br>", unsafe_allow_html=True)

# Timeline Bar Graph for BERT Label:
st.markdown("**Timeline: Months vs BERT Label**")
timeline_bert = df_year.groupby(['month', 'bert_label']).size().reset_index(name='count')
timeline_bert['month'] = pd.Categorical(timeline_bert['month'], categories=months_order, ordered=True)
chart_bert_timeline = alt.Chart(timeline_bert).mark_bar().encode(
    x=alt.X('month:N', title='Month', sort=months_order),
    xOffset='bert_label:N',
    y=alt.Y('count:Q', title='Number of Posts'),
    color='bert_label:N',
    tooltip=['month', 'bert_label', 'count']
).properties(width=700, height=300)
st.altair_chart(chart_bert_timeline, use_container_width=True)

if not timeline_bert.empty:
    ct_bert = timeline_bert.groupby('bert_label')['count'].sum().reset_index()
    max_bert = ct_bert.loc[ct_bert['count'].idxmax()]
    graph_data_bert = f"In {selected_year}, posts with BERT label '{max_bert['bert_label']}' total {max_bert['count']}."
    sample_posts_bert = df_year[df_year["bert_label"] == max_bert['bert_label']].head(5)
    sample_text_bert = " ".join((sample_posts_bert["title"].fillna("") + ". " + sample_posts_bert["body"].fillna("")).tolist())
    try:
        timeline_summary_bert = summarize_with_groq_prompt(sample_text_bert, graph_data_bert, groq_chat)
        st.markdown("**Timeline Summary (BERT):** " + timeline_summary_bert)
    except Exception as e:
        st.markdown("**Timeline Summary (BERT):** Error: " + str(e))
st.markdown("<br>", unsafe_allow_html=True)

# Timeline Bar Graph for Predefined Topic:
st.markdown("**Timeline: Months vs Predefined Topic**")
timeline_topic = df_year.groupby(['month', 'predefined_topic']).size().reset_index(name='count')
timeline_topic['month'] = pd.Categorical(timeline_topic['month'], categories=months_order, ordered=True)
chart_topic_timeline = alt.Chart(timeline_topic).mark_bar().encode(
    x=alt.X('month:N', title='Month', sort=months_order),
    xOffset='predefined_topic:N',
    y=alt.Y('count:Q', title='Number of Posts'),
    color='predefined_topic:N',
    tooltip=['month', 'predefined_topic', 'count']
).properties(width=700, height=300)
st.altair_chart(chart_topic_timeline, use_container_width=True)

if not timeline_topic.empty:
    ct_topic = timeline_topic.groupby('predefined_topic')['count'].sum().reset_index()
    max_topic = ct_topic.loc[ct_topic['count'].idxmax()]
    graph_data_topic = f"In {selected_year}, posts with predefined topic '{max_topic['predefined_topic']}' total {max_topic['count']}."
    sample_posts_topic = df_year[df_year["predefined_topic"] == max_topic['predefined_topic']].head(5)
    sample_text_topic = " ".join((sample_posts_topic["title"].fillna("") + ". " + sample_posts_topic["body"].fillna("")).tolist())
    try:
        timeline_summary_topic = summarize_with_groq_prompt(sample_text_topic, graph_data_topic, groq_chat)
        st.markdown("**Timeline Summary (Predefined Topic):** " + timeline_summary_topic)
    except Exception as e:
        st.markdown("**Timeline Summary (Predefined Topic):** Error: " + str(e))
else:
    st.write("No data available for timeline graphs.")
