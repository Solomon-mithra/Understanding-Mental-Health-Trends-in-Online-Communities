import streamlit as st
import pandas as pd
import altair as alt

# 1. Load Data
def load_data():
    file_path = "merged_results.csv"  # Ensure this file is in the Streamlit app directory
    df = pd.read_csv(file_path)
    return df

df = load_data()

# 2. App Title and Description
st.title("Mental Health Trends in Online Communities")
st.write("Analyzing sentiment and topics from online discussions on mental health.")

# 3. Sidebar Filters
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

# 4. Apply Filters
filtered_df = df.copy()
if topics_selected:
    filtered_df = filtered_df[filtered_df["predefined_topic"].isin(topics_selected)]
if emotions_selected:
    filtered_df = filtered_df[filtered_df["emotion_label"].isin(emotions_selected)]
if bert_labels_selected:
    filtered_df = filtered_df[filtered_df["bert_label"].isin(bert_labels_selected)]

st.write(f"Showing **{len(filtered_df)}** results after applying filters.")

# 5. Display Filtered Data (selected columns only)
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

# 6. Built-In Visualizations: Distribution Charts
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

# 7. Helper function to create Altair heatmaps
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

# 8. Relationship Visualizations with Expanded Summaries and Sample Posts

## 8.1 Emotion Label vs. BERT Label
st.subheader("Relationship: Emotion Label vs. BERT Label")
if not filtered_df.empty:
    # Prepare data for heatmap
    ct_emo_bert = pd.crosstab(filtered_df["emotion_label"], filtered_df["bert_label"]).reset_index()
    ct_emo_bert = ct_emo_bert.melt(id_vars="emotion_label", var_name="bert_label", value_name="count")
    chart_emo_bert = create_heatmap(ct_emo_bert, "bert_label", "emotion_label", "count", "Emotion vs. BERT Label")
    st.altair_chart(chart_emo_bert, use_container_width=True)
    
    # Identify most frequent combination and provide expanded explanation
    ct_orig = pd.crosstab(filtered_df["emotion_label"], filtered_df["bert_label"])
    max_idx = ct_orig.stack().idxmax()  # (emotion, bert)
    max_value = ct_orig.stack().max()
    summary_text = (
        f"The heatmap indicates that posts with emotion label **{max_idx[0]}** are most frequently "
        f"associated with BERT label **{max_idx[1]}** (with {max_value} occurrences). \n\n"
        f"This suggests that the language features associated with the emotion **{max_idx[0]}** "
        f"tend to be recognized by the BERT classifier as **{max_idx[1]}**. This correlation might indicate "
        f"a shared underlying sentiment or thematic element captured by both analysis methods."
    )
    st.markdown("**Summary:** " + summary_text)
    
    # Show sample 5 posts for this combination
    sample_posts = filtered_df[
        (filtered_df["emotion_label"] == max_idx[0]) & (filtered_df["bert_label"] == max_idx[1])
    ].head(5)
    st.markdown("**Sample Posts:**")
    st.write(sample_posts[columns_to_show])
else:
    st.write("No data available for this visualization.")
st.markdown("<br>", unsafe_allow_html=True)

## 8.2 Emotion Label vs. Predefined Topic
st.subheader("Relationship: Emotion Label vs. Predefined Topic")
if not filtered_df.empty:
    ct_emo_topic = pd.crosstab(filtered_df["emotion_label"], filtered_df["predefined_topic"]).reset_index()
    ct_emo_topic = ct_emo_topic.melt(id_vars="emotion_label", var_name="predefined_topic", value_name="count")
    chart_emo_topic = create_heatmap(ct_emo_topic, "predefined_topic", "emotion_label", "count", "Emotion vs. Predefined Topic")
    st.altair_chart(chart_emo_topic, use_container_width=True)
    
    ct_orig = pd.crosstab(filtered_df["emotion_label"], filtered_df["predefined_topic"])
    max_idx = ct_orig.stack().idxmax()  # (emotion, topic)
    max_value = ct_orig.stack().max()
    summary_text = (
        f"The heatmap shows that posts with emotion label **{max_idx[0]}** are most commonly "
        f"associated with predefined topic **{max_idx[1]}** (with {max_value} occurrences). \n\n"
        f"This indicates that when users express **{max_idx[0]}**, they tend to discuss topics related to **{max_idx[1]}**. "
        f"Such a pattern can offer insights into the contexts in which this emotion is prevalent."
    )
    st.markdown("**Summary:** " + summary_text)
    
    sample_posts = filtered_df[
        (filtered_df["emotion_label"] == max_idx[0]) & (filtered_df["predefined_topic"] == max_idx[1])
    ].head(5)
    st.markdown("**Sample Posts:**")
    st.write(sample_posts[columns_to_show])
else:
    st.write("No data available for this visualization.")
st.markdown("<br>", unsafe_allow_html=True)

## 8.3 BERT Label vs. Predefined Topic
st.subheader("Relationship: BERT Label vs. Predefined Topic")
if not filtered_df.empty:
    ct_bert_topic = pd.crosstab(filtered_df["bert_label"], filtered_df["predefined_topic"]).reset_index()
    ct_bert_topic = ct_bert_topic.melt(id_vars="bert_label", var_name="predefined_topic", value_name="count")
    chart_bert_topic = create_heatmap(ct_bert_topic, "predefined_topic", "bert_label", "count", "BERT vs. Predefined Topic")
    st.altair_chart(chart_bert_topic, use_container_width=True)
    
    ct_orig = pd.crosstab(filtered_df["bert_label"], filtered_df["predefined_topic"])
    max_idx = ct_orig.stack().idxmax()  # (bert, topic)
    max_value = ct_orig.stack().max()
    summary_text = (
        f"The heatmap reveals that posts with BERT label **{max_idx[0]}** are most frequently "
        f"linked with predefined topic **{max_idx[1]}** (with {max_value} occurrences). \n\n"
        f"This suggests that the features detected by the BERT classifier for label **{max_idx[0]}** "
        f"are strongly aligned with discussions around **{max_idx[1]}**. This could indicate a thematic consistency in how these posts are characterized."
    )
    st.markdown("**Summary:** " + summary_text)
    
    sample_posts = filtered_df[
        (filtered_df["bert_label"] == max_idx[0]) & (filtered_df["predefined_topic"] == max_idx[1])
    ].head(5)
    st.markdown("**Sample Posts:**")
    st.write(sample_posts[columns_to_show])
else:
    st.write("No data available for this visualization.")
