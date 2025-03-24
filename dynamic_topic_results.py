import os
import pandas as pd
import logging
from transformers import pipeline

# Set up granular logging
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level to capture detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

# Extended list of predefined mental health themes
predefined_labels = [
    "Depression",
    "Anxiety",
    "PTSD",
    "Bipolar Disorder",
    "OCD",
    "Substance Abuse",
    "Eating Disorders",
    "Self-harm",
    "Mental Health Stigma",
    "Therapy/Counseling",
    "Well-being/Resilience",
    "Suicidal Ideation",
    "Trauma & Abuse",
    "Stress & Burnout",
    "Sleep Disorders",
    "Social Isolation",
    "Body Image",
    "Coping Strategies"
]

# Initialize the zero-shot classification pipeline once (will be used for each batch)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

def assign_predefined_label(text, index=None):
    """
    Assigns a predefined mental health label to the input text using zero-shot classification.
    Logs the input snippet, the full classification result (at DEBUG level), and the chosen label.
    """
    try:
        result = classifier(text, candidate_labels=predefined_labels)
        chosen_label = result["labels"][0]
        logging.debug(f"Post {index}: Full classification result: {result}")
        return chosen_label
    except Exception as e:
        logging.error(f"Error classifying post {index}: {e}")
        return "Unclassified"

def process_batch(df_batch, start_index):
    """
    Processes a DataFrame batch (subset of rows) and assigns a predefined label to each post.
    Returns the batch DataFrame with an added 'predefined_topic' column.
    """
    assigned_labels = []
    for i, row in df_batch.iterrows():
        text = row["body"]
        current_index = start_index + i + 1
        logging.info(f"Processing post {current_index}. Text snippet: {text[:100]}...")
        label = assign_predefined_label(text, index=current_index)
        logging.info(f"Assigned label for post {current_index}: {label}")
        assigned_labels.append(label)
    df_batch["predefined_topic"] = assigned_labels
    return df_batch

if __name__ == "__main__":
    input_file = "reddit_mental_health_trends.xlsx"
    output_file = "predefined_topic_results.csv"
    
    logging.info("Loading dataset...")
    df = pd.read_excel(input_file, engine="openpyxl")
    total_rows = len(df)
    logging.info(f"Dataset loaded successfully with {total_rows} rows.")
    
    # Determine how many rows have already been processed
    if os.path.exists(output_file):
        processed_df = pd.read_csv(output_file)
        processed_count = len(processed_df)
        logging.info(f"Output file exists. Already processed {processed_count} rows. Will continue from row {processed_count}.")
    else:
        processed_count = 0
        logging.info("No output file found. Starting from row 0.")
    
    batch_size = 100
    # Process the DataFrame in batches starting from the last processed row
    for start in range(processed_count, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        logging.info(f"Processing batch for rows {start} to {end}...")
        df_batch = df.iloc[start:end].copy()
        df_batch_processed = process_batch(df_batch, start)
        
        # Append the processed batch to the output file
        if start == 0 and processed_count == 0:
            # First batch: write with header
            df_batch_processed.to_csv(output_file, index=False, mode='w')
        else:
            # Append new batches without writing the header again
            df_batch_processed.to_csv(output_file, index=False, mode='a', header=False)
        
        logging.info(f"Batch for rows {start} to {end} processed and appended.")
    
    logging.info("Predefined topic classification completed!")
