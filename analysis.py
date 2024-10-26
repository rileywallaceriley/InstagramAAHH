import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import re
from typing import List, Dict, Optional
import random
import streamlit as st

class InstagramAnalyzer:
    def __init__(self):
        """Initialize the Instagram analyzer."""
        self.competitors_data = {}
        self.hashtag_performance = None
        self.trend_analysis = None
        
    def add_competitor(self, name: str, data: List[Dict]):
        """
        Add a competitor's posting data for analysis.
        
        Parameters:
        name (str): Competitor's name or identifier
        data (list of dict): List of post data containing:
            - posted_at: datetime string
            - caption: str
            - views: int
            - likes: int
            - comments: int
            - shares: int
        """
        df = pd.DataFrame(data)
        df['posted_at'] = pd.to_datetime(df['posted_at'])
        df['day_of_week'] = df['posted_at'].dt.day_name()
        df['hour'] = df['posted_at'].dt.hour
        
        # Extract hashtags from caption
        df['hashtags'] = df['caption'].apply(self._extract_hashtags)
        
        # Calculate engagement score
        df['engagement_score'] = (
            df['views'] * 1 +
            df['likes'] * 2 +
            df['comments'] * 3 +
            df['shares'] * 4
        ) / (df['views'].clip(lower=1))
        
        self.competitors_data[name] = df
        
    def _extract_hashtags(self, caption: str) -> List[str]:
        """Extract hashtags from post caption."""
        if not isinstance(caption, str):
            return []
        # Find all hashtags using regex
        hashtags = re.findall(r'#(\w+)', caption.lower())
        return hashtags
    
    def analyze_posting_times(self) -> Dict:
        """
        Analyze optimal posting times based on engagement.
        
        Returns:
        dict: Analysis of best posting times
        """
        all_posts = []
        for competitor, df in self.competitors_data.items():
            df_copy = df.copy()
            df_copy['competitor'] = competitor
            all_posts.append(df_copy)
            
        if not all_posts:
            return {}
            
        combined_posts = pd.concat(all_posts)
        
        # Analyze by day and hour
        time_analysis = combined_posts.groupby(['day_of_week', 'hour']).agg({
            'engagement_score': ['mean', 'count'],
            'competitor': lambda x: list(set(x))
        }).reset_index()
        
        # Flatten column names
        time_analysis.columns = ['day_of_week', 'hour', 'avg_engagement', 'post_count', 'competitors']
        
        # Filter for statistical significance
        time_analysis = time_analysis[time_analysis['post_count'] >= 3]
        
        # Sort by engagement
        time_analysis = time_analysis.sort_values('avg_engagement', ascending=False)
        
        return {
            'best_times': self._format_best_times(time_analysis.head(10)),
            'daily_patterns': self._analyze_daily_patterns(combined_posts),
            'hourly_patterns': self._analyze_hourly_patterns(combined_posts)
        }
    
    def _format_best_times(self, time_df: pd.DataFrame) -> List[Dict]:
        """Format posting times into readable results."""
        return [
            {
                'day': row['day_of_week'],
                'hour': self._format_hour(row['hour']),
                'avg_engagement': round(row['avg_engagement'], 2),
                'post_count': int(row['post_count']),
                'competitors': row['competitors']
            }
            for _, row in time_df.iterrows()
        ]
    
    def _format_hour(self, hour: int) -> str:
        """Format hour in 12-hour format."""
        if hour == 0:
            return "12 AM"
        elif hour < 12:
            return f"{hour} AM"
        elif hour == 12:
            return "12 PM"
        else:
            return f"{hour-12} PM"
    
    def analyze_hashtags(self, min_usage: int = 3) -> Dict:
        """
        Analyze hashtag performance across all competitors.
        
        Parameters:
        min_usage (int): Minimum number of times a hashtag must be used
        
        Returns:
        dict: Hashtag analysis results
        """
        all_hashtag_data = []
        
        for competitor, df in self.competitors_data.items():
            hashtag_df = df.explode('hashtags')
            hashtag_df = hashtag_df[hashtag_df['hashtags'].notna()]
            hashtag_df['competitor'] = competitor
            all_hashtag_data.append(hashtag_df)
            
        if not all_hashtag_data:
            return {}
            
        combined_hashtags = pd.concat(all_hashtag_data)
        
        # Analyze hashtag performance
        hashtag_stats = combined_hashtags.groupby('hashtags').agg({
            'engagement_score': ['mean', 'count'],
            'views': 'mean',
            'likes': 'mean',
            'comments': 'mean',
            'shares': 'mean',
            'competitor': lambda x: list(set(x))
        }).reset_index()
        
        # Flatten column names
        hashtag_stats.columns = [
            'hashtag', 'avg_engagement', 'usage_count', 'avg_views',
            'avg_likes', 'avg_comments', 'avg_shares', 'used_by'
        ]
        
        # Filter by minimum usage
        hashtag_stats = hashtag_stats[hashtag_stats['usage_count'] >= min_usage]
        
        # Sort by engagement
        hashtag_stats = hashtag_stats.sort_values('avg_engagement', ascending=False)
        
        return {
            'top_hashtags': self._format_hashtag_stats(hashtag_stats.head(20)),
            'hashtag_combinations': self._analyze_hashtag_combinations(combined_hashtags)
        }
    
    def _format_hashtag_stats(self, hashtag_df: pd.DataFrame) -> List[Dict]:
        """Format hashtag statistics into readable results."""
        return [
            {
                'hashtag': f"#{row['hashtag']}",
                'avg_engagement': round(row['avg_engagement'], 2),
                'usage_count': int(row['usage_count']),
                'avg_views': int(row['avg_views']),
                'avg_likes': int(row['avg_likes']),
                'avg_comments': int(row['avg_comments']),
                'avg_shares': int(row['avg_shares']),
                'used_by': row['used_by']
            }
            for _, row in hashtag_df.iterrows()
        ]
    
    def _analyze_hashtag_combinations(self, hashtag_df: pd.DataFrame) -> List[Dict]:
        """Analyze successful hashtag combinations."""
        post_hashtags = hashtag_df.groupby(hashtag_df.index)['hashtags'].agg(list)
        post_engagement = hashtag_df.groupby(hashtag_df.index)['engagement_score'].first()
        
        # Analyze combinations of 2-3 hashtags
        combinations = []
        for idx, hashtags in post_hashtags.items():
            if len(hashtags) >= 2:
                for i in range(2, min(4, len(hashtags) + 1)):
                    for combo in self._get_combinations(hashtags, i):
                        combinations.append({
                            'hashtags': combo,
                            'engagement': post_engagement[idx]
                        })
        
        # Convert to DataFrame for analysis
        if combinations:
            combo_df = pd.DataFrame(combinations)
            combo_stats = combo_df.groupby('hashtags').agg({
                'engagement': ['mean', 'count']
            }).reset_index()
            
            # Filter and sort
            combo_stats = combo_stats[combo_stats[('engagement', 'count')] >= 2]
            combo_stats = combo_stats.sort_values(('engagement', 'mean'), ascending=False)
            
            return [
                {
                    'hashtags': [f"#{tag}" for tag in combo],
                    'avg_engagement': round(stats[('engagement', 'mean')], 2),
                    'usage_count': int(stats[('engagement', 'count')])
                }
                for combo, stats in combo_stats.head(10).iterrows()
            ]
        
        return []
    
    def _get_combinations(self, items: List, r: int) -> List[tuple]:
        """Get all combinations of size r from items."""
        from itertools import combinations
        return list(combinations(items, r))

# Example Streamlit app
def main():
    st.title("Instagram Competitor Analysis")
    
    # File upload for competitor data
    uploaded_file = st.file_uploader("Upload competitor data (CSV)", type="csv")
    
    if uploaded_file is not None:
        # Read competitor data
        data = pd.read_csv(uploaded_file)
        
        # Initialize analyzer
        analyzer = InstagramAnalyzer()
        
        # Add competitor data
        analyzer.add_competitor("Competitor1", data.to_dict('records'))
        
        # Analyze posting times
        st.header("Best Posting Times")
        posting_analysis = analyzer.analyze_posting_times()
        
        for time in posting_analysis['best_times']:
            st.write(f"{time['day']} at {time['hour']}")
            st.write(f"Average Engagement: {time['avg_engagement']}")
            st.write(f"Based on {time['post_count']} posts")
            st.write("---")
        
        # Analyze hashtags
        st.header("Top Performing Hashtags")
        hashtag_analysis = analyzer.analyze_hashtags()
        
        for hashtag in hashtag_analysis['top_hashtags']:
            st.write(f"Hashtag: {hashtag['hashtag']}")
            st.write(f"Average Engagement: {hashtag['avg_engagement']}")
            st.write(f"Used {hashtag['usage_count']} times")
            st.write("---")

if __name__ == "__main__":
    main()