import praw
import pandas as pd
import time
from datetime import datetime

# Replace these with your Reddit API credentials
REDDIT_CLIENT_ID = "TaJw01CqDejvNWyg7fp2KQ"
REDDIT_CLIENT_SECRET = "kALBPtCPn1kb8jxJWmCjDoVs_z5Uxg"
REDDIT_USER_AGENT = "Crazy_Cauliflower591"

reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)

# List of mental health-related subreddits
MENTAL_HEALTH_SUBREDDITS = [
    "mentalhealth", "depression", "Anxiety", "BipolarReddit", "SuicideWatch",
    "mentalillness", "Therapy", "StopSelfHarm", "psychology", "selfimprovement"
]

# Different sorting methods to maximize data collection
SORTING_METHODS = ["new", "top", "hot"]

def fetch_reddit_posts(subreddits=MENTAL_HEALTH_SUBREDDITS, total_limit=100000, batch_size=1000):
    """Fetch Reddit posts from multiple mental health-related subreddits using `.new()`, `.top()`, and `.hot()`."""
    
    all_posts = []
    seen_ids = set()
    
    for subreddit in subreddits:
        for sorting_method in SORTING_METHODS:
            after = None  # Reset pagination for each sorting method
            print(f"\n🔎 Collecting {sorting_method} posts from r/{subreddit}...\n")

            while len(all_posts) < total_limit:
                try:
                    # Fetch new posts with pagination
                    if sorting_method == "new":
                        new_posts = list(reddit.subreddit(subreddit).new(limit=batch_size, params={"after": after}))
                    elif sorting_method == "top":
                        new_posts = list(reddit.subreddit(subreddit).top(limit=batch_size, params={"after": after}))
                    else:  # "hot"
                        new_posts = list(reddit.subreddit(subreddit).hot(limit=batch_size, params={"after": after}))

                    if not new_posts:
                        print(f"⚠️ No more {sorting_method} posts found in r/{subreddit}. Moving to next sorting method.")
                        break  # Stop if there are no more posts

                    for post in new_posts:
                        if post.id not in seen_ids:  # Avoid duplicate posts
                            post_date = datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S')

                            all_posts.append({
                                "id": post.id,
                                "subreddit": f"r/{subreddit}",
                                "sorting_method": sorting_method,
                                "title": post.title,
                                "body": post.selftext if post.selftext else "No content",
                                "date_time": post_date,
                                "url": f"https://www.reddit.com{post.permalink}"
                            })
                            seen_ids.add(post.id)

                    # Update `after` with the last post's fullname to continue pagination
                    after = new_posts[-1].fullname if new_posts else None

                    # Show progress
                    print(f"✅ Collected {len(all_posts)} posts so far...")

                    # Stop if after is None (end of available data)
                    if after is None:
                        print(f"⚠️ No more {sorting_method} posts available in r/{subreddit}. Moving to next sorting method.")
                        break

                    # Reddit API rate limit: Sleep to avoid being blocked
                    time.sleep(2)

                except Exception as e:
                    print(f"⚠️ Error fetching {sorting_method} posts from r/{subreddit}: {e}")
                    break  # Stop in case of an error

    return all_posts

if __name__ == "__main__":
    # Fetch data from multiple subreddits & sorting methods
    reddit_posts = fetch_reddit_posts(total_limit=100000)

    # Convert to DataFrame
    df = pd.DataFrame(reddit_posts)

    if not df.empty:
        # Save as Excel file
        excel_filename = "reddit_mental_health_trends.xlsx"
        df.to_excel(excel_filename, index=False, engine="openpyxl")
        print(f"📂 Data saved to {excel_filename}")
    else:
        print("⚠️ No posts found.")
