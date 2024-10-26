import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import re
from typing import List, Dict, Optional, Set, Tuple
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import random
from itertools import combinations, product
import calendar
from scipy import stats

class InstagramStrategyAnalyzer:
    def __init__(self):
        """Initialize the Instagram strategy analyzer."""
        self.competitors_data = {}
        self.hashtag_performance = None
        self.trend_analysis = None
        self.caption_templates = None
        self.ab_tests = []
        self.content_calendar = None
        
    def add_competitor(self, name: str, data: List[Dict]):
        """Previous competitor data loading code..."""
        # Previous implementation remains the same
        pass
        
    def generate_content_calendar(self, 
                                days: int = 30, 
                                posts_per_day: float = 1.5,
                                content_mix: Optional[Dict[str, float]] = None) -> Dict:
        """
        Generate an optimized content calendar.
        
        Parameters:
        days (int): Number of days to plan
        posts_per_day (float): Average posts per day
        content_mix (dict): Desired content type mix (e.g., {'reels': 0.6, 'posts': 0.4})
        
        Returns:
        dict: Optimized content calendar with recommendations
        """
        if content_mix is None:
            content_mix = {'reels': 0.6, 'posts': 0.3, 'carousels': 0.1}
            
        # Analyze best posting times
        best_times = self._analyze_optimal_posting_times()
        
        # Generate calendar
        start_date = datetime.now()
        calendar_days = []
        total_posts = int(days * posts_per_day)
        
        # Distribute posts across days based on engagement patterns
        daily_distribution = self._calculate_daily_distribution(total_posts, days)
        
        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)
            
            # Get number of posts for this day
            posts_today = daily_distribution[day_offset]
            
            # Get best times for this day
            day_name = current_date.strftime('%A')
            best_day_times = best_times.get(day_name, [])
            
            # Generate posts for the day
            daily_posts = self._generate_daily_posts(
                current_date,
                posts_today,
                best_day_times,
                content_mix
            )
            
            calendar_days.append({
                'date': current_date,
                'posts': daily_posts
            })
        
        self.content_calendar = {
            'calendar': calendar_days,
            'metrics': self._calculate_calendar_metrics(calendar_days),
            'recommendations': self._generate_calendar_recommendations(calendar_days)
        }
        
        return self.content_calendar
    
    def _generate_daily_posts(self, 
                            date: datetime,
                            post_count: int,
                            best_times: List[Dict],
                            content_mix: Dict[str, float]) -> List[Dict]:
        """Generate optimized posts for a specific day."""
        posts = []
        
        # Distribute content types based on mix
        content_types = self._distribute_content_types(post_count, content_mix)
        
        for i, content_type in enumerate(content_types):
            # Get best time slot
            time_slot = best_times[i % len(best_times)] if best_times else {
                'hour': random.randint(9, 20)
            }
            
            # Generate post content
            post = self._generate_post_content(date, time_slot, content_type)
            posts.append(post)
        
        return posts
    
    def _generate_post_content(self, 
                             date: datetime,
                             time_slot: Dict,
                             content_type: str) -> Dict:
        """Generate optimized content for a post."""
        # Get trending topics for the content type
        trending_topics = self._analyze_trending_topics(content_type)
        selected_topic = random.choice(trending_topics)
        
        # Generate caption with A/B testing variants
        caption_variants = self._generate_ab_test_variants(
            content_type,
            selected_topic
        )
        
        return {
            'datetime': date.replace(hour=time_slot['hour']),
            'content_type': content_type,
            'topic': selected_topic,
            'caption_variants': caption_variants,
            'hashtags': self.get_hashtag_recommendations(content_type),
            'estimated_engagement': self._estimate_post_engagement(
                content_type,
                selected_topic,
                time_slot
            )
        }
    
    def generate_ab_tests(self, 
                         content_type: str,
                         test_elements: Optional[List[str]] = None) -> Dict:
        """
        Generate A/B testing recommendations.
        
        Parameters:
        content_type (str): Type of content to test
        test_elements (list): Elements to test (e.g., ['caption_style', 'hashtag_count'])
        
        Returns:
        dict: A/B testing plan with variants
        """
        if test_elements is None:
            test_elements = ['caption_style', 'hashtag_count', 'posting_time']
            
        test_plan = {}
        
        for element in test_elements:
            variants = self._generate_test_variants(element, content_type)
            test_plan[element] = {
                'variants': variants,
                'success_metrics': self._define_success_metrics(element),
                'test_duration': self._calculate_test_duration(element),
                'sample_size': self._calculate_required_sample_size(element)
            }
            
        self.ab_tests.append({
            'content_type': content_type,
            'plan': test_plan,
            'start_date': datetime.now()
        })
        
        return test_plan
    
    def _generate_test_variants(self, 
                              element: str, 
                              content_type: str) -> List[Dict]:
        """Generate variants for A/B testing."""
        variants = []
        
        if element == 'caption_style':
            styles = ['engaging', 'professional', 'educational', 'conversational']
            for style in styles:
                caption = self.generate_optimized_caption(
                    content_type=content_type,
                    topic="test topic",
                    tone=style
                )
                variants.append({
                    'style': style,
                    'example': caption['caption'],
                    'expected_engagement': caption['metrics']['estimated_engagement']
                })
                
        elif element == 'hashtag_count':
            counts = [5, 10, 15, 20]
            for count in counts:
                hashtags = self.get_hashtag_recommendations(
                    content_type,
                    max_hashtags=count
                )
                variants.append({
                    'count': count,
                    'example': hashtags['primary_hashtags'][:count],
                    'expected_engagement': self._estimate_hashtag_performance(
                        hashtags['primary_hashtags'][:count]
                    )
                })
                
        elif element == 'posting_time':
            times = self._analyze_optimal_posting_times()
            for day, time_slots in times.items():
                for slot in time_slots[:2]:  # Top 2 times per day
                    variants.append({
                        'day': day,
                        'time': slot['hour'],
                        'expected_engagement': slot['engagement_score']
                    })
                    
        return variants
    
    def analyze_competitive_gaps(self) -> Dict:
        """
        Analyze competitive gaps and opportunities.
        
        Returns:
        dict: Competitive analysis and recommendations
        """
        gaps = {
            'content_gaps': self._analyze_content_gaps(),
            'timing_gaps': self._analyze_timing_gaps(),
            'hashtag_gaps': self._analyze_hashtag_gaps(),
            'engagement_gaps': self._analyze_engagement_gaps(),
            'opportunities': self._generate_opportunity_recommendations()
        }
        
        return gaps
    
    def _analyze_content_gaps(self) -> Dict:
        """Analyze gaps in content coverage."""
        content_analysis = defaultdict(lambda: defaultdict(int))
        
        # Analyze competitor content
        for competitor, df in self.competitors_data.items():
            for content_type in df['content_type'].unique():
                content_analysis[competitor][content_type] = len(
                    df[df['content_type'] == content_type]
                )
                
        # Identify underserved content types
        gaps = []
        for content_type in set().union(
            *[types.keys() for types in content_analysis.values()]
        ):
            avg_posts = np.mean([
                analysis[content_type] 
                for analysis in content_analysis.values()
            ])
            if avg_posts < 5:  # Threshold for gap identification
                gaps.append({
                    'content_type': content_type,
                    'average_posts': avg_posts,
                    'opportunity_score': self._calculate_opportunity_score(
                        content_type, avg_posts
                    )
                })
                
        return {
            'gaps': gaps,
            'recommendations': self._generate_content_recommendations(gaps)
        }
    
    def _calculate_opportunity_score(self, 
                                  content_type: str, 
                                  current_coverage: float) -> float:
        """Calculate opportunity score for a content gap."""
        # Base score from lack of coverage
        base_score = 1 - (current_coverage / 10)  # Normalize to 0-1
        
        # Adjust for engagement potential
        engagement_multiplier = self._get_content_type_engagement(content_type)
        
        # Adjust for trend alignment
        trend_multiplier = self._get_trend_alignment_score(content_type)
        
        return base_score * engagement_multiplier * trend_multiplier
    
    def _generate_content_recommendations(self, gaps: List[Dict]) -> List[Dict]:
        """Generate specific recommendations for content gaps."""
        recommendations = []
        
        for gap in gaps:
            content_type = gap['content_type']
            opportunity_score = gap['opportunity_score']
            
            # Generate specific recommendation
            recommendation = {
                'content_type': content_type,
                'opportunity_score': opportunity_score,
                'suggested_frequency': self._calculate_suggested_frequency(
                    opportunity_score
                ),
                'content_ideas': self._generate_content_ideas(content_type),
                'hashtag_strategy': self.get_hashtag_recommendations(content_type),
                'best_posting_times': self._get_best_times_for_content(content_type)
            }
            
            recommendations.append(recommendation)
            
        return recommendations

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = InstagramStrategyAnalyzer()
    
    # Generate content calendar
    calendar = analyzer.generate_content_calendar(
        days=30,
        posts_per_day=1.5,
        content_mix={'reels': 0.6, 'posts': 0.3, 'carousels': 0.1}
    )
    
    # Generate A/B tests
    ab_tests = analyzer.generate_ab_tests(
        content_type="reel",
        test_elements=['caption_style', 'hashtag_count', 'posting_time']
    )
    
    # Analyze competitive gaps
    gaps = analyzer.analyze_competitive_gaps()
    
    # Print example results
    print("\nContent Calendar Preview:")
    for day in calendar['calendar'][:3]:  # Show first 3 days
        print(f"\nDate: {day['date'].strftime('%Y-%m-%d')}")
        for post in day['posts']:
            print(f"  - {post['content_type']} at {post['datetime'].strftime('%H:%M')}")
            print(f"    Topic: {post['topic']}")
            print(f"    Est. Engagement: {post['estimated_engagement']}")
            
    print("\nA/B Test Recommendations:")
    for element, tests in ab_tests.items():
        print(f"\n{element}:")
        for variant in tests['variants'][:2]:  # Show first 2 variants
            print(f"  - Variant: {variant}")
            
    print("\nCompetitive Gaps:")
    for gap_type, analysis in gaps.items():
        print(f"\n{gap_type}:")
        if isinstance(analysis, dict) and 'recommendations' in analysis:
            for rec in analysis['recommendations'][:2]:  # Show first 2 recommendations
                print(f"  - {rec}")
