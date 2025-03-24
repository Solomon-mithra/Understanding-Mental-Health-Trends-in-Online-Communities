import pandas as pd
import nltk
import torch
import logging
import os
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline, AutoTokenizer

# Download VADER lexicon
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load sentiment analysis models
bert_sentiment = pipeline("sentiment-analysis", device=0 if device == "cuda" else -1)
goemotions = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=0 if device == "cuda" else -1)

# Load tokenizer for DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Define output file
output_file = "sentiment_results.csv"

# Remove existing file to start fresh
if os.path.exists(output_file):
    os.remove(output_file)
    logging.info("Existing results file removed. Starting fresh.")

def analyze_sentiment_vader(text):
    """Perform sentiment analysis using VADER."""
    if not isinstance(text, str) or text.strip() == "":
        return None
    return sia.polarity_scores(text)['compound']

def analyze_sentiment_bert(text, index, total):
    """Perform sentiment analysis using BERT with strict truncation for individual texts."""
    if not isinstance(text, str) or text.strip() == "":
        return None, None, None, None
    
    # Tokenize and strictly truncate to 512 tokens
    tokenized_input = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    truncated_text = tokenizer.decode(tokenized_input['input_ids'][0], skip_special_tokens=True)
    
    logging.info(f"Processing post {index+1}/{total}: {truncated_text[:100]}...")  # Log first 100 characters for visibility
    try:
        sentiment_result = bert_sentiment(truncated_text)[0]
    except Exception as e:
        logging.error(f"BERT Sentiment Analysis Failed for post {index+1}: {e}")
        sentiment_result = {"label": "error", "score": 0.0}
    
    try:
        emotion_result = goemotions(truncated_text)[0]
    except Exception as e:
        logging.error(f"GoEmotions Analysis Failed for post {index+1}: {e}")
        emotion_result = {"label": "error", "score": 0.0}
    
    logging.info(f"Post {index+1} - BERT Sentiment: {sentiment_result}, GoEmotions: {emotion_result}")
    return sentiment_result['label'], sentiment_result['score'], emotion_result['label'], emotion_result['score']

def process_sentiment(data, batch_size=1000):
    """Apply sentiment and emotion analysis on the dataset individually per row in batches."""
    logging.info("Starting sentiment analysis...")
    total_rows = len(data)
    processed_rows = 0
    
    for i in range(0, total_rows, batch_size):
        batch = data.iloc[i:i+batch_size].copy()
        batch['vader_sentiment'] = batch['body'].apply(analyze_sentiment_vader)
        
        # Process each row individually with BERT and GoEmotions
        results = [analyze_sentiment_bert(text, idx + i, total_rows) for idx, text in enumerate(batch['body'])]
        batch['bert_label'], batch['bert_score'], batch['emotion_label'], batch['emotion_score'] = zip(*results)
        
        # Append results to file
        batch.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)
        processed_rows += len(batch)
        logging.info(f"Processed {processed_rows}/{total_rows} posts. Appended batch to file.")
    
    logging.info("Sentiment analysis completed.")

if __name__ == "__main__":
    logging.info("Loading dataset...")
    df = pd.read_excel("reddit_mental_health_trends.xlsx", engine="openpyxl")  # Corrected file reading
    logging.info(f"Dataset loaded successfully with {len(df)} rows.")
    
    process_sentiment(df)
    
    logging.info("Sentiment analysis results saved to 'sentiment_results.csv'.")
    print("Sentiment analysis completed!")
