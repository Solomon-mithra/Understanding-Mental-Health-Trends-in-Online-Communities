import pandas as pd

# File paths for the CSV files
sentiment_file = "sentiment_results.csv"
topic_file = "predefined_topic_results.csv"

# Load the CSV files into DataFrames
sentiment_df = pd.read_csv(sentiment_file)
topic_df = pd.read_csv(topic_file)

# Define the common key column name
common_key = "id"

# Check if the common key exists in both DataFrames
if common_key in sentiment_df.columns and common_key in topic_df.columns:
    # Identify duplicate columns in topic_df (other than the key)
    duplicate_cols = [col for col in topic_df.columns if col in sentiment_df.columns and col != common_key]
    # Drop duplicate columns from topic_df
    topic_df_unique = topic_df.drop(columns=duplicate_cols)
    
    # Merge the DataFrames on the common key without repeating columns
    merged_df = pd.merge(sentiment_df, topic_df_unique, on=common_key, how="inner")
    
    print(f"Data merged successfully on '{common_key}'!")
    print("Merged DataFrame shape:", merged_df.shape)
    
    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv("merged_results.csv", index=False)
    print("Merged data saved to 'merged_results.csv'.")
else:
    print(f"Error: The common key '{common_key}' was not found in both CSV files. Please verify the column names.")
