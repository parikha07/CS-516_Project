import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
RANDOM_SEED = 99992
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
faker = Faker()
Faker.seed(RANDOM_SEED)

class ProtectedAttributes:
    """Class to handle protected attributes and fairness metrics"""
    
    def __init__(self):
        # Define protected attributes
        self.protected_attributes = [
            'age_group',
            'gender',
            'ethnicity',
            'socioeconomic_status',
            'disability_status'
        ]
        
        # Define values for protected attributes
        self.attribute_values = {
            'age_group': ['18-24', '25-34', '35-44', '45-54', '55-64', '65+'],
            'gender': ['Male', 'Female', 'Non-binary'],
            'ethnicity': ['White', 'Black', 'Hispanic', 'Asian', 'Indigenous', 'Other'],
            'socioeconomic_status': ['Low', 'Medium', 'High'],
            'disability_status': ['None', 'Physical', 'Cognitive', 'Sensory', 'Multiple']
        }
        
        # Define default distributions for protected attributes
        self.default_distributions = {
            'age_group': {'18-24': 0.15, '25-34': 0.25, '35-44': 0.25, '45-54': 0.15, '55-64': 0.1, '65+': 0.1},
            'gender': {'Male': 0.48, 'Female': 0.48, 'Non-binary': 0.04},
            'ethnicity': {'White': 0.6, 'Black': 0.13, 'Hispanic': 0.18, 'Asian': 0.06, 'Indigenous': 0.01, 'Other': 0.02},
            'socioeconomic_status': {'Low': 0.3, 'Medium': 0.5, 'High': 0.2},
            'disability_status': {'None': 0.8, 'Physical': 0.08, 'Cognitive': 0.05, 'Sensory': 0.05, 'Multiple': 0.02}
        }
    
    def normalize_distribution(self, distribution):
        """Normalize a distribution to ensure probabilities sum to 1"""
        total = sum(distribution.values())
        if abs(total - 1.0) > 0.0001:
            return {k: v/total for k, v in distribution.items()}
        return distribution
    
    def generate_protected_attributes(self, num_users, custom_distributions=None):
        """Generate protected attributes for users"""
        protected_data = {}
        
        # Use custom distributions if provided, otherwise use defaults
        distributions = self.default_distributions.copy()
        if custom_distributions:
            for attr, dist in custom_distributions.items():
                if attr in distributions:
                    distributions[attr] = dist
                    distributions[attr] = self.normalize_distribution(distributions[attr])
        
        # Generate attributes for each user
        for attr in self.protected_attributes:
            values = list(distributions[attr].keys())
            probabilities = list(distributions[attr].values())
            protected_data[attr] = np.random.choice(values, size=num_users, p=probabilities)
        
        return pd.DataFrame(protected_data)
    
    def calculate_fairness_metrics(self, data, target_column):
        """Calculate fairness metrics for the dataset"""
        metrics = {}
        
        for attr in self.protected_attributes:
            # Skip if attribute not in data
            if attr not in data.columns:
                continue
                
            # Demographic parity
            overall_rate = data[target_column].mean()
            group_rates = data.groupby(attr)[target_column].mean()
            max_disparity = group_rates.max() - group_rates.min()
            
            metrics[f"{attr}_disparity"] = max_disparity
            metrics[f"{attr}_group_rates"] = group_rates.to_dict()
            
            # Calculate disparate impact for binary outcomes
            if len(data[target_column].unique()) <= 2:
                # Find group with highest and lowest rates
                highest_group = group_rates.idxmax()
                lowest_group = group_rates.idxmin()
                
                # Avoid division by zero
                if group_rates[lowest_group] > 0:
                    disparate_impact = group_rates[highest_group] / group_rates[lowest_group]
                else:
                    disparate_impact = float('inf')
                    
                metrics[f"{attr}_disparate_impact"] = disparate_impact
        
        return metrics

class SyntheticAdDataGenerator:
    """A class to generate synthetic data for ad targeting research with controllable bias parameters."""
    
    def __init__(self, 
                num_users=10000, 
                num_ads=1000,
                avg_impressions_per_user=20,
                bias_level=0.7,
                click_base_rate=0.05,
                include_enhanced_features=True,
                output_directory=None,
                random_seed=RANDOM_SEED,
                custom_distributions=None):
        """Initialize the data generator with specified parameters."""
        self.num_users = num_users
        self.num_ads = num_ads
        self.avg_impressions_per_user = avg_impressions_per_user
        self.bias_level = bias_level
        self.click_base_rate = click_base_rate
        self.include_enhanced_features = include_enhanced_features
        self.output_directory = output_directory
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        self.faker = Faker()
        self.faker.seed_instance(random_seed)
        
        # Initialize protected attributes handler
        self.protected_attrs = ProtectedAttributes()
        self.custom_distributions = custom_distributions
        
        # Define demographic options
        self.init_demographic_options()
        
        # Define bias parameters with default values
        self.bias_params = {
            'age_bias_strength': 1.0,
            'gender_bias_strength': 1.0,
            'ethnicity_bias_strength': 1.0,
            'socioeconomic_bias_strength': 1.0,
            'disability_bias_strength': 1.0,
            'behavioral_bias_strength': 1.0,
            'geographic_bias_strength': 1.0,
            'device_bias_strength': 1.0,
            'temporal_bias_strength': 1.0,
            'seasonal_bias_strength': 1.0,
            'weekday_bias_strength': 1.0,
            'time_of_day_bias_strength': 1.0,
            'ad_fatigue_strength': 1.0
        }
        
        # Initialize distributions
        self.init_distributions()
    
    def set_bias_parameters(self, **kwargs):
        """Set bias parameters"""
        for param, value in kwargs.items():
            if param in self.bias_params:
                self.bias_params[param] = value
    
    def init_demographic_options(self):
        """Initialize demographic options"""
        # Education and income used for socioeconomic status
        self.education_levels = ['High School', 'Some College', 'Bachelor\'s', 'Master\'s', 'PhD']
        self.income_brackets = ['$0-$25k', '$25k-$50k', '$50k-$75k', '$75k-$100k', '$100k+']
        
        # Location options
        self.locations = ['Urban', 'Suburban', 'Rural']
        self.relationship_statuses = ['Single', 'In Relationship', 'Married', 'Divorced', 'Widowed']
        self.occupations = [
            'Student', 'Professional', 'Technical', 'Service', 'Administrative', 
            'Sales', 'Management', 'Retired', 'Unemployed', 'Other'
        ]
        self.interests = [
            'Technology', 'Sports', 'Travel', 'Food', 'Fashion',
            'Gaming', 'Arts', 'Health', 'Education', 'Finance'
        ]
        
        # Geographic options
        self.us_regions = ['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West']
        self.us_states = [
            # Northeast
            'ME', 'NH', 'VT', 'MA', 'RI', 'CT', 'NY', 'NJ', 'PA',
            # Southeast
            'DE', 'MD', 'DC', 'VA', 'WV', 'NC', 'SC', 'GA', 'FL', 'KY', 'TN', 'AL', 'MS', 'AR', 'LA',
            # Midwest
            'OH', 'IN', 'IL', 'MI', 'WI', 'MN', 'IA', 'MO', 'ND', 'SD', 'NE', 'KS',
            # Southwest
            'OK', 'TX', 'NM', 'AZ',
            # West
            'MT', 'ID', 'WY', 'CO', 'UT', 'NV', 'WA', 'OR', 'CA', 'AK', 'HI'
        ]
        self.location_details = ['Urban Core', 'Inner Suburb', 'Outer Suburb', 'Small City', 'Town', 'Rural']
        self.population_density = ['Very High', 'High', 'Medium', 'Low', 'Very Low']
        
        # Device characteristics
        self.operating_systems = ['iOS', 'Android', 'Windows', 'MacOS', 'ChromeOS', 'Linux']
        self.browsers = ['Chrome', 'Safari', 'Firefox', 'Edge', 'Samsung Browser', 'Opera']
        self.connection_types = ['5G', '4G', 'WiFi', 'Fiber', 'Cable', 'DSL', 'Satellite']
        self.screen_sizes = ['Small', 'Medium', 'Large', 'Extra Large']
        
        # Temporal patterns
        self.seasons = ['Winter', 'Spring', 'Summer', 'Fall']
        self.weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.hours_of_day = list(range(24))
        
        # Season to month mapping
        self.month_to_season = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
            11: 'Fall', 12: 'Winter'
        }
        
        # Ad content types
        self.ad_categories = [
            'Technology', 'Finance', 'Healthcare', 'Education', 
            'Professional Services', 'Retail', 'Entertainment',
            'Automotive', 'Travel', 'Home Services'
        ]
        
        # Job ad types
        self.job_categories = [
            'Executive Leadership', 'Middle Management', 'Entry Level',
            'Technical', 'Administrative', 'Creative', 'Sales',
            'Healthcare', 'Education', 'Service Industry'
        ]
        
        # Ad formats
        self.ad_formats = ['Image', 'Video', 'Carousel', 'Text', 'Interactive']
        self.ad_objectives = ['Awareness', 'Consideration', 'Conversion', 'Engagement', 'Traffic']
        
        # Temporal affinities
        self.init_temporal_affinities()
    
    def init_temporal_affinities(self):
        """Initialize temporal affinity data"""
        # Seasonal category affinities
        self.seasonal_category_affinity = {
            'Winter': {
                'Travel': 1.5,  # Winter vacations
                'Retail': 1.8,  # Holiday shopping
                'Healthcare': 1.3,  # Cold/flu season
            },
            'Spring': {
                'Home Services': 1.4,  # Spring cleaning
                'Retail': 1.2,  # Spring fashion
                'Healthcare': 1.2,  # Allergy season
            },
            'Summer': {
                'Travel': 1.8,  # Summer vacations
                'Entertainment': 1.4,  # Summer blockbusters
                'Retail': 1.3,  # Summer sales
            },
            'Fall': {
                'Education': 1.6,  # Back to school
                'Professional Services': 1.3,  # Business restart
                'Retail': 1.4,  # Fall shopping
            }
        }
        
        # Day of week affinities
        self.weekday_category_affinity = {
            'Monday': {'Professional Services': 1.3, 'Education': 1.2},
            'Tuesday': {'Technology': 1.2, 'Finance': 1.2},
            'Wednesday': {'Retail': 1.1, 'Entertainment': 1.1},
            'Thursday': {'Travel': 1.2, 'Finance': 1.1},
            'Friday': {'Entertainment': 1.4, 'Travel': 1.3},
            'Saturday': {'Retail': 1.5, 'Entertainment': 1.4},
            'Sunday': {'Retail': 1.3, 'Travel': 1.2}
        }
        
        # Time of day affinities (based on hour of day)
        self.hour_category_affinity = {
            # Morning (6-11)
            6: {'Finance': 1.2, 'Education': 1.2},
            7: {'Finance': 1.3, 'Education': 1.3},
            8: {'Professional Services': 1.4, 'Finance': 1.4},
            9: {'Professional Services': 1.5, 'Technology': 1.3},
            10: {'Technology': 1.3, 'Retail': 1.2},
            11: {'Retail': 1.3, 'Professional Services': 1.2},
            # Afternoon (12-17)
            12: {'Retail': 1.4, 'Travel': 1.3},
            13: {'Retail': 1.4, 'Entertainment': 1.2},
            14: {'Technology': 1.2, 'Entertainment': 1.3},
            15: {'Technology': 1.2, 'Retail': 1.3},
            16: {'Entertainment': 1.4, 'Retail': 1.4},
            17: {'Entertainment': 1.5, 'Automotive': 1.3},
            # Evening (18-23)
            18: {'Entertainment': 1.6, 'Retail': 1.4},
            19: {'Entertainment': 1.7, 'Home Services': 1.3},
            20: {'Entertainment': 1.8, 'Travel': 1.4},
            21: {'Entertainment': 1.7, 'Travel': 1.5},
            22: {'Entertainment': 1.6, 'Technology': 1.4},
            23: {'Entertainment': 1.5, 'Technology': 1.3},
            # Late Night (0-5)
            0: {'Entertainment': 1.4, 'Technology': 1.3},
            1: {'Entertainment': 1.3, 'Technology': 1.2},
            2: {'Entertainment': 1.2, 'Technology': 1.1},
            3: {'Technology': 1.1, 'Education': 1.1},
            4: {'Education': 1.2, 'Finance': 1.1},
            5: {'Education': 1.3, 'Finance': 1.1}
        }
    
    def init_distributions(self):
        """Initialize distributions for user demographics and other features"""
        # User distribution parameters (set defaults)
        self.user_distribution = {
            'education': {'High School': 0.25, 'Some College': 0.3, 'Bachelor\'s': 0.3, 'Master\'s': 0.1, 'PhD': 0.05},
            'income': {'$0-$25k': 0.2, '$25k-$50k': 0.3, '$50k-$75k': 0.25, '$75k-$100k': 0.15, '$100k+': 0.1},
            'location': {'Urban': 0.4, 'Suburban': 0.4, 'Rural': 0.2},
            'relationship_status': {'Single': 0.3, 'In Relationship': 0.15, 'Married': 0.4, 'Divorced': 0.1, 'Widowed': 0.05},
            'occupation': {'Student': 0.1, 'Professional': 0.2, 'Technical': 0.15, 'Service': 0.15, 'Administrative': 0.1, 
                      'Sales': 0.1, 'Management': 0.1, 'Retired': 0.05, 'Unemployed': 0.03, 'Other': 0.02}
        }
        
        # Define the geo distributions
        self.geo_distribution = {
            'region': {'Northeast': 0.18, 'Southeast': 0.25, 'Midwest': 0.21, 'Southwest': 0.12, 'West': 0.24},
            'state': self.create_state_distribution(),
            'location_detail': {'Urban Core': 0.2, 'Inner Suburb': 0.3, 'Outer Suburb': 0.25, 'Small City': 0.1, 'Town': 0.1, 'Rural': 0.05},
            'population_density': {'Very High': 0.15, 'High': 0.25, 'Medium': 0.3, 'Low': 0.2, 'Very Low': 0.1}
        }
        
        self.device_distribution = {
            'operating_system': {'iOS': 0.35, 'Android': 0.35, 'Windows': 0.2, 'MacOS': 0.07, 'ChromeOS': 0.02, 'Linux': 0.01},
            'browser': {'Chrome': 0.45, 'Safari': 0.3, 'Firefox': 0.08, 'Edge': 0.12, 'Samsung Browser': 0.04, 'Opera': 0.01},
            'connection_type': {'5G': 0.15, '4G': 0.3, 'WiFi': 0.35, 'Fiber': 0.08, 'Cable': 0.07, 'DSL': 0.03, 'Satellite': 0.02},
            'screen_size': {'Small': 0.35, 'Medium': 0.45, 'Large': 0.15, 'Extra Large': 0.05}
        }
        
        self.behavior_distribution = {
            'browse_depth': {'Low (1-2 pages)': 0.3, 'Medium (3-5 pages)': 0.5, 'High (6+ pages)': 0.2},
            'session_duration': {'Short (< 1 min)': 0.25, 'Medium (1-5 min)': 0.5, 'Long (> 5 min)': 0.25},
            'return_visitor': {'New': 0.3, 'Returning': 0.5, 'Loyal': 0.2},
            'customer_lifetime_value': {'Low': 0.6, 'Medium': 0.25, 'High': 0.1, 'VIP': 0.05}
        }
        
        # Normalize all distributions
        self.normalize_all_distributions()
    
    def create_state_distribution(self):
        """Create state distribution based on US population"""
        return {
            'CA': 0.12, 'TX': 0.09, 'FL': 0.07, 'NY': 0.06, 'PA': 0.04, 'IL': 0.04, 'OH': 0.035,
            'GA': 0.032, 'NC': 0.031, 'MI': 0.03, 'NJ': 0.026, 'VA': 0.025, 'WA': 0.023, 'AZ': 0.022,
            'TN': 0.021, 'MA': 0.02, 'IN': 0.02, 'MO': 0.018, 'MD': 0.018, 'WI': 0.018, 'MN': 0.017,
            'CO': 0.018, 'SC': 0.015, 'AL': 0.015, 'LA': 0.014, 'KY': 0.013, 'OR': 0.013, 'OK': 0.012,
            'CT': 0.011, 'UT': 0.01, 'MS': 0.009, 'AR': 0.009, 'KS': 0.009, 'NV': 0.009, 'IA': 0.009,
            'NM': 0.006, 'NE': 0.006, 'ID': 0.006, 'WV': 0.005, 'HI': 0.004, 'NH': 0.004, 'ME': 0.004,
            'MT': 0.003, 'RI': 0.003, 'DE': 0.003, 'SD': 0.003, 'ND': 0.002, 'AK': 0.002, 'VT': 0.002,
            'WY': 0.002, 'DC': 0.002
        }
    
    def normalize_all_distributions(self):
        """Normalize all distributions to ensure probabilities sum to 1"""
        # Normalize user_distribution
        for key, dist in self.user_distribution.items():
            total = sum(dist.values())
            if abs(total - 1.0) > 0.0001:
                print(f"Normalizing {key} distribution: sum was {total}")
                self.user_distribution[key] = {k: v/total for k, v in dist.items()}
        
        # Normalize geo_distribution
        for key, dist in self.geo_distribution.items():
            total = sum(dist.values())
            if abs(total - 1.0) > 0.0001:
                print(f"Normalizing {key} distribution: sum was {total}")
                self.geo_distribution[key] = {k: v/total for k, v in dist.items()}
        
        # Normalize device_distribution
        for key, dist in self.device_distribution.items():
            total = sum(dist.values())
            if abs(total - 1.0) > 0.0001:
                print(f"Normalizing {key} distribution: sum was {total}")
                self.device_distribution[key] = {k: v/total for k, v in dist.items()}
        
        # Normalize behavior_distribution
        for key, dist in self.behavior_distribution.items():
            total = sum(dist.values())
            if abs(total - 1.0) > 0.0001:
                print(f"Normalizing {key} distribution: sum was {total}")
                self.behavior_distribution[key] = {k: v/total for k, v in dist.items()}
    
    def generate_social_network(self):
        """Generate a social network connecting users"""
        print("Generating social network...")
        users = list(range(self.num_users))
        network = {}
        
        # For each user, generate connections
        for user_id in tqdm(users, desc="Creating user connections"):
            # Number of connections follows a power law distribution
            num_connections = min(
                int(np.random.power(2) * 20) + 1,  # Power law distribution
                self.num_users - 1  # Can't have more connections than other users
            )
            
            # Select connections, prioritizing similar demographics
            potential_connections = [u for u in users if u != user_id]
            network[user_id] = set(np.random.choice(
                potential_connections, 
                size=min(num_connections, len(potential_connections)),
                replace=False
            ))
        
        return network
    
    def add_social_influence(self, user_df, impressions_df, network=None):
        """Add social influence effects to impressions"""
        print("Adding social influence effects...")
        if network is None:
            network = self.generate_social_network()
        
        # Create a copy to avoid modifying the original
        influenced_impressions = impressions_df.copy()
        
        # Track which ads each user has clicked
        user_clicks = {}
        for _, imp in tqdm(impressions_df[impressions_df['clicked'] == 1].iterrows(), 
                         desc="Processing user clicks", 
                         total=len(impressions_df[impressions_df['clicked'] == 1])):
            user_id = imp['user_id']
            ad_id = imp['ad_id']
            
            if user_id not in user_clicks:
                user_clicks[user_id] = set()
            user_clicks[user_id].add(ad_id)
        
        # Apply social influence to each impression
        for idx, imp in tqdm(influenced_impressions.iterrows(), 
                            desc="Applying social influence", 
                            total=len(influenced_impressions)):
            user_id = imp['user_id']
            ad_id = imp['ad_id']
            
            # Skip if already clicked
            if imp['clicked'] == 1:
                continue
            
            # Check if friends clicked this ad
            friends = network.get(user_id, set())
            friend_clicks = sum(1 for friend in friends if 
                              friend in user_clicks and ad_id in user_clicks[friend])
            
            # Calculate social influence factor (diminishing returns)
            influence_factor = min(1.0, 0.1 * friend_clicks)
            
            # Apply influence to click probability
            if influence_factor > 0:
                base_click_prob = self.click_base_rate
                social_click_prob = base_click_prob * (1 + influence_factor)
                # Re-roll for click with social influence
                clicked = np.random.binomial(1, min(1.0, social_click_prob))
                influenced_impressions.at[idx, 'clicked'] = clicked
        
        return influenced_impressions
    
    def generate_user_demographics(self):
        """Generate user demographic information with configurable distributions"""
        print("Generating user demographics...")
        start_time = time.time()
        
        # Generate protected attributes first
        protected_df = self.protected_attrs.generate_protected_attributes(
            self.num_users, self.custom_distributions
        )
        
        # Generate other demographic features
        users = []
        
        for i in tqdm(range(self.num_users), desc="Creating user profiles"):
            # Start with protected attributes
            user = {
                'user_id': i,
                'age_group': protected_df['age_group'].iloc[i],
                'gender': protected_df['gender'].iloc[i],
                'ethnicity': protected_df['ethnicity'].iloc[i],
                'socioeconomic_status': protected_df['socioeconomic_status'].iloc[i],
                'disability_status': protected_df['disability_status'].iloc[i]
            }
            
            # Add standard demographic attributes
            user.update({
                'education': np.random.choice(
                    self.education_levels, 
                    p=list(self.user_distribution['education'].values())
                ),
                'income': np.random.choice(
                    self.income_brackets, 
                    p=list(self.user_distribution['income'].values())
                ),
                'location': np.random.choice(
                    self.locations, 
                    p=list(self.user_distribution['location'].values())
                ),
            })
            
            # Add US geographic attributes
            user.update({
                'region': np.random.choice(
                    self.us_regions,
                    p=list(self.geo_distribution['region'].values())
                ),
                'state': np.random.choice(
                    self.us_states,
                    p=list(self.geo_distribution['state'].values())
                ),
                'location_detail': np.random.choice(
                    self.location_details,
                    p=list(self.geo_distribution['location_detail'].values())
                ),
                'population_density': np.random.choice(
                    self.population_density,
                    p=list(self.geo_distribution['population_density'].values())
                )
            })
            
            # Add device characteristics
            user.update({
                'primary_os': np.random.choice(
                    self.operating_systems,
                    p=list(self.device_distribution['operating_system'].values())
                ),
                'primary_browser': np.random.choice(
                    self.browsers,
                    p=list(self.device_distribution['browser'].values())
                ),
                'connection_type': np.random.choice(
                    self.connection_types,
                    p=list(self.device_distribution['connection_type'].values())
                ),
                'screen_size': np.random.choice(
                    self.screen_sizes,
                    p=list(self.device_distribution['screen_size'].values())
                )
            })
            
            # Add browsing behavior metrics
            user.update({
                'browse_depth': np.random.choice(
                    list(self.behavior_distribution['browse_depth'].keys()),
                    p=list(self.behavior_distribution['browse_depth'].values())
                ),
                'session_duration': np.random.choice(
                    list(self.behavior_distribution['session_duration'].keys()),
                    p=list(self.behavior_distribution['session_duration'].values())
                ),
                'visitor_status': np.random.choice(
                    list(self.behavior_distribution['return_visitor'].keys()),
                    p=list(self.behavior_distribution['return_visitor'].values())
                ),
                'customer_value': np.random.choice(
                    list(self.behavior_distribution['customer_lifetime_value'].keys()),
                    p=list(self.behavior_distribution['customer_lifetime_value'].values())
                )
            })
            
            # Add enhanced demographic attributes if enabled
            if self.include_enhanced_features:
                user.update({
                    'relationship_status': np.random.choice(
                        self.relationship_statuses, 
                        p=list(self.user_distribution['relationship_status'].values())
                    ),
                    'occupation': np.random.choice(
                        self.occupations, 
                        p=list(self.user_distribution['occupation'].values())
                    ),
                })
                    
            # Generate interest scores for different categories
            for interest in self.interests:
                user[f'interest_{interest.lower()}'] = np.random.beta(1.2, 3.0)
            
            # Browsing behavior - always included
            user.update({
                'browsing_tech': np.random.beta(2, 5),
                'browsing_finance': np.random.beta(2, 5),
                'browsing_healthcare': np.random.beta(2, 5),
                'browsing_education': np.random.beta(2, 5),
                'browsing_professional': np.random.beta(2, 5),
                'browsing_shopping': np.random.beta(2, 5),
                'browsing_entertainment': np.random.beta(2, 5),
                'activity_morning': np.random.beta(2, 2),
                'activity_afternoon': np.random.beta(2, 2),
                'activity_evening': np.random.beta(2, 2),
                'activity_night': np.random.beta(2, 2),
            })
            
            # Add correlation between demographics and browsing behavior
            if user['age_group'] in ['18-24', '25-34']:
                user['browsing_tech'] = min(1.0, user['browsing_tech'] * 1.5)
                user['browsing_entertainment'] = min(1.0, user['browsing_entertainment'] * 1.5)
            
            if user['age_group'] in ['55-64', '65+']:
                user['browsing_healthcare'] = min(1.0, user['browsing_healthcare'] * 1.5)
                user['browsing_finance'] = min(1.0, user['browsing_finance'] * 1.5)
            
            # Add correlations based on protected attributes
            if user['disability_status'] != 'None':
                user['browsing_healthcare'] = min(1.0, user['browsing_healthcare'] * 1.8)
            
            if user['socioeconomic_status'] == 'High':
                user['browsing_finance'] = min(1.0, user['browsing_finance'] * 1.5)
                user['browsing_travel'] = min(1.0, user.get('browsing_travel', 0) * 1.5)
            
            # Add numeric metrics for visitor engagement
            user['pages_per_session'] = {
                'Low (1-2 pages)': np.random.randint(1, 3),
                'Medium (3-5 pages)': np.random.randint(3, 6),
                'High (6+ pages)': np.random.randint(6, 15)
            }[user['browse_depth']]
            
            user['seconds_per_session'] = {
                'Short (< 1 min)': np.random.randint(5, 60),
                'Medium (1-5 min)': np.random.randint(60, 300),
                'Long (> 5 min)': np.random.randint(300, 1200)
            }[user['session_duration']]
            
            users.append(user)
        
        elapsed = time.time() - start_time
        print(f"Generated {self.num_users} users in {elapsed:.2f} seconds.")
        return pd.DataFrame(users)
    
    def _apply_bias_based_on_protected_attributes(self, user_df, ad_df):
        """Apply targeting bias based on protected attributes"""
        # Create bias scores for ads
        ad_df['protected_attr_score'] = np.random.uniform(0, 1, len(ad_df))
        
        # Initialize targeting weight columns for protected attributes
        for attr in self.protected_attrs.protected_attributes:
            for value in self.protected_attrs.attribute_values[attr]:
                col_name = f"{attr}_{value}_targeting_weight"
                ad_df[col_name] = 0.5  # Neutral starting point
        
        # Apply biases to each ad
        for idx, ad in ad_df.iterrows():
            # Different biases for different ad categories
            if ad['is_job_listing'] == 1:
                # Job listing biases
                if ad['job_category'] in ['Executive Leadership', 'Technical']:
                    # Apply gender bias
                    bias_factor = ad['protected_attr_score'] * self.bias_level * self.bias_params['gender_bias_strength']
                    ad_df.at[idx, 'gender_Male_targeting_weight'] = 0.5 + (bias_factor * 0.5)
                    ad_df.at[idx, 'gender_Female_targeting_weight'] = 0.5 - (bias_factor * 0.5)
                    ad_df.at[idx, 'gender_Non-binary_targeting_weight'] = 0.2
                    
                    # Apply age bias
                    age_bias_factor = ad['protected_attr_score'] * self.bias_level * self.bias_params['age_bias_strength']
                    ad_df.at[idx, 'age_group_18-24_targeting_weight'] = 0.5 - (age_bias_factor * 0.3)
                    ad_df.at[idx, 'age_group_25-34_targeting_weight'] = 0.5 + (age_bias_factor * 0.4)
                    ad_df.at[idx, 'age_group_35-44_targeting_weight'] = 0.5 + (age_bias_factor * 0.4)
                    ad_df.at[idx, 'age_group_45-54_targeting_weight'] = 0.5 + (age_bias_factor * 0.2)
                    ad_df.at[idx, 'age_group_55-64_targeting_weight'] = 0.5 - (age_bias_factor * 0.3)
                    ad_df.at[idx, 'age_group_65+_targeting_weight'] = 0.5 - (age_bias_factor * 0.4)
                    
                    # Apply disability bias
                    disability_bias_factor = ad['protected_attr_score'] * self.bias_level * self.bias_params['disability_bias_strength']
                    ad_df.at[idx, 'disability_status_None_targeting_weight'] = 0.5 + (disability_bias_factor * 0.5)
                    for disability_type in ['Physical', 'Cognitive', 'Sensory', 'Multiple']:
                        ad_df.at[idx, f'disability_status_{disability_type}_targeting_weight'] = 0.5 - (disability_bias_factor * 0.5)
                
                elif ad['job_category'] in ['Administrative', 'Education', 'Healthcare']:
                    # Reverse gender bias
                    bias_factor = ad['protected_attr_score'] * self.bias_level * self.bias_params['gender_bias_strength']
                    ad_df.at[idx, 'gender_Male_targeting_weight'] = 0.5 - (bias_factor * 0.5)
                    ad_df.at[idx, 'gender_Female_targeting_weight'] = 0.5 + (bias_factor * 0.5)
                    ad_df.at[idx, 'gender_Non-binary_targeting_weight'] = 0.2
            else:
                # Regular product/service ad biases
                if ad['category'] in ['Technology', 'Finance', 'Automotive']:
                    # Apply gender bias
                    bias_factor = ad['protected_attr_score'] * self.bias_level * self.bias_params['gender_bias_strength']
                    ad_df.at[idx, 'gender_Male_targeting_weight'] = 0.5 + (bias_factor * 0.3)
                    ad_df.at[idx, 'gender_Female_targeting_weight'] = 0.5 - (bias_factor * 0.3)
                    ad_df.at[idx, 'gender_Non-binary_targeting_weight'] = 0.3
                    
                elif ad['category'] in ['Healthcare', 'Retail', 'Home Services']:
                    # Reverse gender bias
                    bias_factor = ad['protected_attr_score'] * self.bias_level * self.bias_params['gender_bias_strength']
                    ad_df.at[idx, 'gender_Male_targeting_weight'] = 0.5 - (bias_factor * 0.3)
                    ad_df.at[idx, 'gender_Female_targeting_weight'] = 0.5 + (bias_factor * 0.3)
                    ad_df.at[idx, 'gender_Non-binary_targeting_weight'] = 0.3
                
                # Apply socioeconomic status bias for luxury categories
                if ad['category'] in ['Finance', 'Travel', 'Automotive']:
                    socio_bias_factor = ad['protected_attr_score'] * self.bias_level * self.bias_params['socioeconomic_bias_strength']
                    ad_df.at[idx, 'socioeconomic_status_High_targeting_weight'] = 0.5 + (socio_bias_factor * 0.5)
                    ad_df.at[idx, 'socioeconomic_status_Medium_targeting_weight'] = 0.5
                    ad_df.at[idx, 'socioeconomic_status_Low_targeting_weight'] = 0.5 - (socio_bias_factor * 0.5)
                
                # Apply ethnic bias (for example purposes only - demonstrating effect)
                ethnicity_bias_factor = ad['protected_attr_score'] * self.bias_level * self.bias_params['ethnicity_bias_strength']
                if ad['category'] in ['Finance', 'Technology']:
                    for ethnicity in ['White', 'Asian']:
                        ad_df.at[idx, f'ethnicity_{ethnicity}_targeting_weight'] = 0.5 + (ethnicity_bias_factor * 0.3)
                    for ethnicity in ['Black', 'Hispanic', 'Indigenous', 'Other']:
                        ad_df.at[idx, f'ethnicity_{ethnicity}_targeting_weight'] = 0.5 - (ethnicity_bias_factor * 0.3)
        
        return user_df, ad_df
    
    def _apply_temporal_bias(self, user_df, ad_df, timestamp=None):
        """Apply temporal bias (season, day of week, hour of day)"""
        # If no timestamp provided, use current time
        if timestamp is None:
            timestamp = datetime.now()
        
        # Extract temporal components
        month = timestamp.month
        day_of_week = timestamp.strftime('%A')
        hour = timestamp.hour
        season = self.month_to_season[month]
        
        # Add temporal targeting weights to ads
        ad_df['temporal_score'] = np.random.uniform(0, 1, len(ad_df))
        ad_df['seasonal_weight'] = 1.0
        ad_df['weekday_weight'] = 1.0
        ad_df['time_of_day_weight'] = 1.0
        
        # Apply seasonal bias
        for idx, ad in ad_df.iterrows():
            # Apply seasonal affinity
            category = ad['category']
            season_multiplier = self.seasonal_category_affinity.get(season, {}).get(category, 1.0)
            seasonal_bias = (season_multiplier - 1.0) * self.bias_level * self.bias_params['seasonal_bias_strength']
            ad_df.at[idx, 'seasonal_weight'] = 1.0 + seasonal_bias
            
            # Apply day of week affinity
            weekday_multiplier = self.weekday_category_affinity.get(day_of_week, {}).get(category, 1.0)
            weekday_bias = (weekday_multiplier - 1.0) * self.bias_level * self.bias_params['weekday_bias_strength']
            ad_df.at[idx, 'weekday_weight'] = 1.0 + weekday_bias
            
            # Apply time of day affinity
            hour_multiplier = self.hour_category_affinity.get(hour, {}).get(category, 1.0)
            hour_bias = (hour_multiplier - 1.0) * self.bias_level * self.bias_params['time_of_day_bias_strength']
            ad_df.at[idx, 'time_of_day_weight'] = 1.0 + hour_bias
        
        return user_df, ad_df
    
    def _apply_geographic_bias(self, user_df, ad_df):
        """Apply geographic bias to ad targeting"""
        # Create geographic targeting columns in ad dataframe
        ad_df['geo_score'] = np.random.uniform(0, 1, len(ad_df))
        
        # Initialize region targeting weights
        for region in self.us_regions:
            ad_df[f'{region}_targeting_weight'] = 0.5
        
        # For each ad, apply geographic targeting bias
        for idx, ad in ad_df.iterrows():
            # Regional bias based on ad category
            if ad['category'] in ['Finance', 'Professional Services']:
                # These categories target more in Northeast and West
                ad_df.at[idx, 'Northeast_targeting_weight'] = 0.5 + (0.5 * ad['geo_score'] * self.bias_level * self.bias_params['geographic_bias_strength'])
                ad_df.at[idx, 'West_targeting_weight'] = 0.5 + (0.4 * ad['geo_score'] * self.bias_level * self.bias_params['geographic_bias_strength'])
            
            elif ad['category'] in ['Automotive', 'Retail']:
                # These categories target more in Midwest and South
                ad_df.at[idx, 'Midwest_targeting_weight'] = 0.5 + (0.4 * ad['geo_score'] * self.bias_level * self.bias_params['geographic_bias_strength'])
                ad_df.at[idx, 'Southeast_targeting_weight'] = 0.5 + (0.4 * ad['geo_score'] * self.bias_level * self.bias_params['geographic_bias_strength'])
                ad_df.at[idx, 'Southwest_targeting_weight'] = 0.5 + (0.3 * ad['geo_score'] * self.bias_level * self.bias_params['geographic_bias_strength'])
        
        return user_df, ad_df

    def _apply_device_bias(self, user_df, ad_df):
        """Apply device-based bias to ad targeting"""
        # Add device targeting columns
        ad_df['device_score'] = np.random.uniform(0, 1, len(ad_df))
        
        # Initialize device targeting weights
        ad_df['mobile_targeting_weight'] = 0.5
        ad_df['desktop_targeting_weight'] = 0.5
        ad_df['apple_targeting_weight'] = 0.5
        ad_df['android_targeting_weight'] = 0.5
        
        # For each ad, apply device targeting bias
        for idx, ad in ad_df.iterrows():
            if ad['category'] in ['Gaming', 'Technology', 'Entertainment']:
                # These favor mobile
                ad_df.at[idx, 'mobile_targeting_weight'] = 0.5 + (0.4 * ad['device_score'] * self.bias_level * self.bias_params['device_bias_strength'])
                
                # Some tech ads target Apple users more
                if ad['category'] == 'Technology' and np.random.random() < 0.7:
                    ad_df.at[idx, 'apple_targeting_weight'] = 0.5 + (0.5 * ad['device_score'] * self.bias_level * self.bias_params['device_bias_strength'])
                
            elif ad['category'] in ['Professional Services', 'Finance']:
                # These favor desktop
                ad_df.at[idx, 'desktop_targeting_weight'] = 0.5 + (0.4 * ad['device_score'] * self.bias_level * self.bias_params['device_bias_strength'])
        
        return user_df, ad_df

    def _apply_behavior_bias(self, user_df, ad_df):
        """Apply browsing behavior bias to ad targeting"""
        # Add behavior targeting columns
        ad_df['behavior_score'] = np.random.uniform(0, 1, len(ad_df))
        
        # Initialize behavior targeting weights
        ad_df['high_engagement_weight'] = 0.5
        ad_df['high_value_weight'] = 0.5
        ad_df['loyal_visitor_weight'] = 0.5
        
        # For each ad, apply behavior targeting bias
        for idx, ad in ad_df.iterrows():
            if ad['category'] in ['Finance', 'Automotive', 'Travel']:
                # High-value ads target engaged users and those with high customer value
                ad_df.at[idx, 'high_engagement_weight'] = 0.5 + (0.5 * ad['behavior_score'] * self.bias_level * self.bias_params['behavioral_bias_strength'])
                ad_df.at[idx, 'high_value_weight'] = 0.5 + (0.6 * ad['behavior_score'] * self.bias_level * self.bias_params['behavioral_bias_strength'])
            
            # Loyalty program and subscription ads target loyal visitors
            if 'subscription' in str(ad['ad_title']).lower() or 'loyalty' in str(ad['ad_title']).lower():
                ad_df.at[idx, 'loyal_visitor_weight'] = 0.5 + (0.7 * ad['behavior_score'] * self.bias_level * self.bias_params['behavioral_bias_strength'])
        
        return user_df, ad_df
    
    def generate_ads(self, num_ads=None):
        """Generate a set of ads with various attributes"""
        print("Generating ads...")
        start_time = time.time()
        if num_ads is None:
            num_ads = self.num_ads
            
        ads = []
        
        for i in tqdm(range(num_ads), desc="Creating ads"):
            # Determine if this is a job listing ad
            is_job_listing = np.random.binomial(1, 0.3)  # 30% chance of job ad
            
            if is_job_listing:
                category = 'Professional Services'  # All job listings are in professional services
                job_category = np.random.choice(self.job_categories)
                ad_title = f"{job_category} Position at {self.faker.company()}"
                compensation = np.random.choice(['Low', 'Medium', 'High', 'Very High'], p=[0.3, 0.4, 0.2, 0.1])
            else:
                category = np.random.choice(self.ad_categories)
                job_category = None
                ad_title = f"{self.faker.company()} {category} {np.random.choice(['Product', 'Service', 'Offer'])}"
                compensation = None
            
            # Base ad attributes
            ad = {
                'ad_id': i,
                'advertiser': self.faker.company(),
                'category': category,
                'ad_title': ad_title,
                'ad_description': self.faker.text(max_nb_chars=200),
                'is_job_listing': is_job_listing,
                'job_category': job_category,
                'compensation': compensation,
                'creation_date': self.faker.date_between(start_date='-90d', end_date='today'),
                'budget': np.random.choice(['Low', 'Medium', 'High'], p=[0.3, 0.5, 0.2]),
                'targeting_strategy': np.random.choice(['Broad', 'Specific'], p=[0.4, 0.6]),
            }
            
            # Add enhanced ad features if enabled
            if self.include_enhanced_features:
                ad.update({
                    'ad_format': np.random.choice(self.ad_formats),
                    'ad_objective': np.random.choice(self.ad_objectives),
                    'target_interests': ','.join(np.random.choice(
                        self.interests, size=np.random.randint(1, 4), replace=False
                    )),
                })
            
            ads.append(ad)
            
        elapsed = time.time() - start_time
        print(f"Generated {num_ads} ads in {elapsed:.2f} seconds.")
        return pd.DataFrame(ads)
    
    def generate_impressions_with_fatigue(self, user_df, ad_df, avg_impressions_per_user=None, days=30):
        """Generate impressions with ad fatigue effect over multiple days"""
        print(f"Generating impressions over {days} days with ad fatigue...")
        all_impressions = []
        
        # Use instance value if not specified
        if avg_impressions_per_user is None:
            avg_impressions_per_user = self.avg_impressions_per_user
        
        # Calculate daily impressions
        daily_impressions_per_user = avg_impressions_per_user / days
        
        # Track ad exposure count for each user
        user_ad_exposure = {user_id: {} for user_id in user_df['user_id']}
        
        # Generate impressions for each day
        for day in tqdm(range(days), desc="Generating daily impressions"):
            # Create timestamp for this day
            timestamp = datetime.now() - timedelta(days=(days-day-1))
            
            # Apply temporal bias for this day
            user_df_temp, ad_df_temp = self._apply_temporal_bias(user_df.copy(), ad_df.copy(), timestamp)
            
            # Generate daily impressions
            daily_impressions = self.generate_ad_impressions(
                user_df_temp, ad_df_temp, 
                avg_impressions_per_user=daily_impressions_per_user
            )
            
            # Apply fatigue effect based on previous exposure
            fatigue_impressions = []
            for _, imp in daily_impressions.iterrows():
                user_id = imp['user_id']
                ad_id = imp['ad_id']
                
                # Increase exposure count
                if ad_id not in user_ad_exposure[user_id]:
                    user_ad_exposure[user_id][ad_id] = 0
                user_ad_exposure[user_id][ad_id] += 1
                
                # Calculate fatigue factor
                exposure_count = user_ad_exposure[user_id][ad_id]
                fatigue_factor = max(0.5, 1.0 - (exposure_count * 0.1 * self.bias_params['ad_fatigue_strength']))
                
                # Apply fatigue to click probability
                if imp['clicked'] == 1:
                    # Recompute click with fatigue
                    base_click_prob = self.click_base_rate
                    fatigued_click_prob = base_click_prob * fatigue_factor
                    # Re-roll for click with fatigue
                    clicked = np.random.binomial(1, fatigued_click_prob)
                    imp = imp.copy()
                    imp['clicked'] = clicked
                
                fatigue_impressions.append(imp)
            
            # Add day field to impressions
            daily_df = pd.DataFrame(fatigue_impressions)
            if not daily_df.empty:
                daily_df['day'] = day
                daily_df['date'] = timestamp.date()
                all_impressions.append(daily_df)
        
        # Combine all days of impressions
        if all_impressions:
            return pd.concat(all_impressions, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def generate_ad_impressions(self, user_df, ad_df, avg_impressions_per_user=None):
        """Generate ad impression data based on users and ads"""
        print("Generating ad impressions...")
        start_time = time.time()
        
        # Use instance value if not specified
        if avg_impressions_per_user is None:
            avg_impressions_per_user = self.avg_impressions_per_user
            
        # First apply all bias modifiers to our ad dataframe
        print("Applying bias factors...")
        bias_functions = [
            self._apply_bias_based_on_protected_attributes,
            self._apply_geographic_bias,
            self._apply_device_bias,
            self._apply_behavior_bias
        ]
        
        for bias_func in tqdm(bias_functions, desc="Applying bias modifiers"):
            user_df, ad_df = bias_func(user_df, ad_df)
        
        # Now generate impressions
        impressions = []
        
        for user_idx, user in tqdm(user_df.iterrows(), total=len(user_df), desc="Processing users"):
            # Determine how many ads this user sees (poisson distribution around the average)
            num_impressions = np.random.poisson(avg_impressions_per_user)
            
            # Generate ad selection probabilities for this user
            ad_probs = []
            
            for ad_idx, ad in ad_df.iterrows():
                # Base probability
                base_prob = 1.0
                
                # Apply protected attribute targeting weights
                for attr in self.protected_attrs.protected_attributes:
                    attr_value = user[attr]
                    weight_col = f"{attr}_{attr_value}_targeting_weight"
                    if weight_col in ad:
                        attr_weight = ad[weight_col]
                        base_prob *= attr_weight
                
                # Apply geographic targeting weight
                region_weight_key = f"{user['region']}_targeting_weight"
                if region_weight_key in ad:
                    region_weight = ad[region_weight_key]
                    base_prob *= region_weight
                
                # Apply device targeting weight
                if user['primary_os'] in ['iOS', 'MacOS']:
                    base_prob *= ad['apple_targeting_weight']
                elif user['primary_os'] == 'Android':
                    base_prob *= ad['android_targeting_weight']
                    
                if user['screen_size'] in ['Small', 'Medium']:
                    base_prob *= ad['mobile_targeting_weight']
                else:
                    base_prob *= ad['desktop_targeting_weight']
                    
                # Apply behavior targeting weight
                if user['browse_depth'] == 'High (6+ pages)' or user['session_duration'] == 'Long (> 5 min)':
                    base_prob *= ad['high_engagement_weight']
                    
                if user['customer_value'] in ['High', 'VIP']:
                    base_prob *= ad['high_value_weight']
                    
                if user['visitor_status'] == 'Loyal':
                    base_prob *= ad['loyal_visitor_weight']
                
                # Apply temporal weights if present
                for temp_weight in ['seasonal_weight', 'weekday_weight', 'time_of_day_weight']:
                    if temp_weight in ad:
                        base_prob *= ad[temp_weight]
                
                ad_probs.append(base_prob)
            
            # Normalize probabilities
            if sum(ad_probs) > 0:
                ad_probs = [p/sum(ad_probs) for p in ad_probs]
            else:
                ad_probs = [1/len(ad_probs) for _ in range(len(ad_probs))]
            
            # Select ads based on probabilities
            shown_ad_indices = np.random.choice(
                len(ad_df), 
                size=min(num_impressions, len(ad_df)), 
                replace=False, 
                p=ad_probs
            )
            
            # Generate impression data
            for ad_idx in shown_ad_indices:
                ad = ad_df.iloc[ad_idx]
                
                # Determine if user clicked
                click_prob = self.click_base_rate  # Base click probability
                
                # Adjust click probability based on matching factors
                if ad['is_job_listing'] == 1:
                    # For job listings, click probability is influenced by match with user profile
                    if ad['job_category'] in ['Executive Leadership', 'Technical'] and user['gender'] == 'Male':
                        click_prob *= 1.5
                    elif ad['job_category'] in ['Administrative', 'Healthcare'] and user['gender'] == 'Female':
                        click_prob *= 1.5
                
                # Determine if clicked
                clicked = np.random.binomial(1, click_prob)
                
                # Generate timestamp
                timestamp = self.faker.date_time_between(start_date='-30d', end_date='now')
                
                impression = {
                    'impression_id': len(impressions),
                    'user_id': user['user_id'],
                    'ad_id': ad['ad_id'],
                    'timestamp': timestamp,
                    'clicked': clicked,
                    'device': 'mobile' if user['screen_size'] in ['Small', 'Medium'] else 'desktop',
                    'platform': np.random.choice(['social_media', 'search_engine', 'website', 'app']),
                    'operating_system': user['primary_os'],
                    'browser': user['primary_browser'],
                    'connection_type': user['connection_type'],
                    'pages_in_session': user['pages_per_session'],
                    'session_duration_seconds': user['seconds_per_session'],
                    'visitor_status': user['visitor_status']
                }
                impressions.append(impression)
        
        elapsed = time.time() - start_time
        print(f"Generated {len(impressions)} impressions in {elapsed:.2f} seconds.")
        return pd.DataFrame(impressions)
    
    def generate_conversion_data(self, users, ads, impressions):
        """Generate conversion data for clicked impressions"""
        print("Generating conversion data...")
        start_time = time.time()
        
        # Extract clicks
        clicks = impressions[impressions['clicked'] == 1].copy()
        
        if clicks.empty:
            return pd.DataFrame()
        
        conversions = []
        
        for _, click in tqdm(clicks.iterrows(), total=len(clicks), desc="Processing clicks"):
            # Base conversion probability
            conversion_prob = 0.1
            
            # Adjust conversion probability based on user/ad attributes
            # Extract user and ad data for this click
            user = users[users['user_id'] == click['user_id']].iloc[0]
            ad = ads[ads['ad_id'] == click['ad_id']].iloc[0]
            
            # Higher income users convert better on luxury items
            if user['socioeconomic_status'] == 'High' and ad['category'] in ['Automotive', 'Travel', 'Finance']:
                conversion_prob *= 1.5
            
            # Interest-matching improves conversion
            if 'target_interests' in ad and hasattr(user, 'interest_technology') and 'Technology' in str(ad['target_interests']) and user['interest_technology'] > 0.7:
                conversion_prob *= 1.3
                
            # Disability bias - lower conversion for certain categories
            if user['disability_status'] != 'None' and ad['category'] in ['Travel', 'Automotive']:
                conversion_prob *= 0.7
            
            # Determine conversion type based on ad category
            if ad['is_job_listing'] == 1:
                conversion_types = ['Application Started', 'Application Completed', 'Interview Scheduled']
                conversion_values = [0, 0, 0]  # No direct monetary value for job applications
            elif ad['category'] in ['Retail', 'Travel', 'Automotive']:
                conversion_types = ['Add to Cart', 'Purchase', 'Upsell']
                conversion_values = [
                    np.random.uniform(10, 50),
                    np.random.uniform(50, 500),
                    np.random.uniform(20, 200)
                ]
            else:
                conversion_types = ['Sign Up', 'Free Trial', 'Subscription']
                conversion_values = [0, 0, np.random.uniform(5, 50)]
            
            # Determine if conversion happens and what type
            if np.random.random() < conversion_prob:
                conversion_type_idx = np.random.choice(
                    range(len(conversion_types)), 
                    p=[0.6, 0.3, 0.1]  # Most conversions are the first type
                )
                conversion_type = conversion_types[conversion_type_idx]
                conversion_value = conversion_values[conversion_type_idx]
                
                # Generate timestamp (slightly after click)
                time_delay = timedelta(minutes=np.random.randint(1, 30))
                conversion_timestamp = click['timestamp'] + time_delay
                
                conversion = {
                    'conversion_id': len(conversions),
                    'impression_id': click['impression_id'],
                    'user_id': click['user_id'],
                    'ad_id': click['ad_id'],
                    'conversion_type': conversion_type,
                    'conversion_value': conversion_value,
                    'timestamp': conversion_timestamp,
                    'device': click['device'],
                    'platform': click['platform']
                }
                conversions.append(conversion)
        
        elapsed = time.time() - start_time
        print(f"Generated {len(conversions)} conversions in {elapsed:.2f} seconds.")
        return pd.DataFrame(conversions) if conversions else pd.DataFrame()
    
    def generate_complete_dataset(self, with_temporal=False, with_social=False):
        """Generate a complete dataset with users, ads, and impressions"""
        print("Generating user demographics...")
        users = self.generate_user_demographics()
        
        print("Generating ads...")
        ads = self.generate_ads()
        
        print("Generating ad impressions with bias level:", self.bias_level)
        
        if with_temporal:
            impressions = self.generate_impressions_with_fatigue(users, ads)
        else:
            impressions = self.generate_ad_impressions(users, ads)
            
        if with_social:
            print("Adding social network effects...")
            impressions = self.add_social_influence(users, impressions)
            
        print("Generating conversion data...")
        conversions = self.generate_conversion_data(users, ads, impressions)
        
        return {
            'users': users,
            'ads': ads,
            'impressions': impressions,
            'conversions': conversions
        }
    
    def analyze_bias(self, dataset):
        """Analyze the dataset to quantify bias and fairness metrics"""
        users = dataset['users']
        ads = dataset['ads']
        impressions = dataset['impressions']
        
        # Join impressions with users and ads
        analysis_df = impressions.merge(users, on='user_id').merge(ads, on='ad_id')
        
        # Calculate fairness metrics for protected attributes
        fairness_metrics = self.protected_attrs.calculate_fairness_metrics(analysis_df, 'clicked')
        
        print("\n==== Fairness Metrics for Protected Attributes ====")
        for metric, value in fairness_metrics.items():
            if 'group_rates' not in metric:
                print(f"{metric}: {value}")
        
        # Get job ads for separate analysis
        job_ads = analysis_df[analysis_df['is_job_listing'] == 1]
        
        # Gender analysis for job listings
        if not job_ads.empty:
            print("\n==== Gender Bias Analysis for Job Listings ====")
            gender_job_counts = job_ads.groupby(['gender', 'job_category']).size().unstack(fill_value=0)
            gender_job_proportions = gender_job_counts.div(gender_job_counts.sum(axis=1), axis=0)
            
            print("\nProportion of job ad impressions by gender and job category:")
            print(gender_job_proportions)
            
            # Calculate disparity metrics
            leadership_jobs = ['Executive Leadership', 'Middle Management']
            leadership_exposure = job_ads[job_ads['job_category'].isin(leadership_jobs)].groupby('gender').size() / job_ads.groupby('gender').size()
            
            print("\nExposure to leadership job ads by gender:")
            print(leadership_exposure)
            
            # Job category by gender heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(gender_job_proportions, annot=True, cmap="YlGnBu", fmt=".2f")
            plt.title("Proportion of Job Ad Impressions by Gender and Category")
            plt.tight_layout()
            plt.savefig("gender_job_heatmap.png")
        
        # Disability status analysis
        print("\n==== Disability Status Analysis ====")
        disability_category_counts = analysis_df.groupby(['disability_status', 'category']).size().unstack(fill_value=0)
        disability_category_proportions = disability_category_counts.div(disability_category_counts.sum(axis=1), axis=0)
        
        print("\nProportion of ad impressions by disability status and category:")
        print(disability_category_proportions)
        
        # Socioeconomic status analysis
        print("\n==== Socioeconomic Status Analysis ====")
        socio_category_counts = analysis_df.groupby(['socioeconomic_status', 'category']).size().unstack(fill_value=0)
        socio_category_proportions = socio_category_counts.div(socio_category_counts.sum(axis=1), axis=0)
        
        print("\nProportion of ad impressions by socioeconomic status and category:")
        print(socio_category_proportions)
        
        # Click-through rate analysis
        print("\n==== Click-through Rate Analysis by Protected Attributes ====")
        
        for attr in self.protected_attrs.protected_attributes:
            ctr_by_attr = analysis_df.groupby(attr)['clicked'].mean()
            print(f"\nClick-through rate by {attr}:")
            print(ctr_by_attr)
            
            # Visualize
            plt.figure(figsize=(10, 6))
            ctr_by_attr.plot(kind='bar')
            plt.title(f'Click-Through Rate by {attr}')
            plt.ylim(0, max(0.2, ctr_by_attr.max() * 1.2))
            plt.tight_layout()
            plt.savefig(f"ctr_by_{attr}.png")
        
        # Temporal analysis if available
        if 'day' in impressions.columns:
            print("\n==== Temporal Analysis ====")
            ctr_by_day = impressions.groupby('day')['clicked'].mean()
            
            print("\nClick-through rate over time (ad fatigue effect):")
            print(ctr_by_day)
            
            plt.figure(figsize=(12, 6))
            ctr_by_day.plot(kind='line', marker='o')
            plt.title('Click-Through Rate Over Time (Ad Fatigue Effect)')
            plt.xlabel('Day')
            plt.ylabel('CTR')
            plt.tight_layout()
            plt.savefig('temporal_ctr.png')
        
        # Return metrics dictionary for further analysis
        return fairness_metrics
    
    def create_visualization_dashboard(self, dataset):
        """Create visualization dashboard for the dataset"""
        users = dataset['users']
        ads = dataset['ads']
        impressions = dataset['impressions']
        
        # Create a directory for visualizations
        os.makedirs('visualizations', exist_ok=True)
        
        # 1. Protected attribute distributions
        for attr in self.protected_attrs.protected_attributes:
            plt.figure(figsize=(10, 6))
            users[attr].value_counts().plot(kind='bar')
            plt.title(f'User {attr.replace("_", " ").title()} Distribution')
            plt.tight_layout()
            plt.savefig(f'visualizations/{attr}_distribution.png')
        
        # 2. Ad category distribution
        plt.figure(figsize=(12, 6))
        ads['category'].value_counts().plot(kind='bar')
        plt.title('Ad Category Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('visualizations/ad_categories.png')
        
        # 3. Bias analysis for protected attributes
        analysis_df = impressions.merge(users, on='user_id').merge(ads, on='ad_id')
        
        # Generate heatmaps for click rates by protected attributes and ad categories
        for attr in self.protected_attrs.protected_attributes:
            plt.figure(figsize=(14, 10))
            attr_category_ctr = analysis_df.groupby([attr, 'category'])['clicked'].mean().unstack()
            sns.heatmap(attr_category_ctr, annot=True, cmap='YlGnBu', fmt='.3f')
            plt.title(f'Click-Through Rate by {attr.replace("_", " ").title()} and Ad Category')
            plt.tight_layout()
            plt.savefig(f'visualizations/{attr}_category_heatmap.png')
        
        # 4. Device analysis
        plt.figure(figsize=(10, 6))
        device_ctr = analysis_df.groupby('device')['clicked'].mean()
        device_ctr.plot(kind='bar')
        plt.title('Click-Through Rate by Device Type')
        plt.tight_layout()
        plt.savefig('visualizations/device_ctr.png')
        
        # 5. Regional analysis
        plt.figure(figsize=(12, 8))
        region_ctr = analysis_df.groupby('region')['clicked'].mean()
        region_ctr.plot(kind='bar')
        plt.title('Click-Through Rate by Region')
        plt.tight_layout()
        plt.savefig('visualizations/region_ctr.png')
        
        # 6. Time-based analysis if available
        if 'day' in impressions.columns:
            plt.figure(figsize=(12, 6))
            day_ctr = impressions.groupby('day')['clicked'].mean()
            day_ctr.plot(kind='line', marker='o')
            plt.title('Click-Through Rate Over Time (Ad Fatigue Effect)')
            plt.xlabel('Day')
            plt.ylabel('CTR')
            plt.tight_layout()
            plt.savefig('visualizations/temporal_ctr.png')
            
            # Analyze CTR by day for different protected attributes
            for attr in self.protected_attrs.protected_attributes:
                temp_analysis = impressions.merge(users[['user_id', attr]], on='user_id')
                plt.figure(figsize=(12, 8))
                for value in users[attr].unique():
                    attr_data = temp_analysis[temp_analysis[attr] == value]
                    if not attr_data.empty:
                        day_values = attr_data.groupby('day')['clicked'].mean()
                        plt.plot(day_values.index, day_values.values, marker='o', label=value)
                
                plt.title(f'Click-Through Rate by Day for Different {attr.replace("_", " ").title()} Values')
                plt.xlabel('Day')
                plt.ylabel('CTR')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'visualizations/temporal_ctr_by_{attr}.png')
        
        print("Visualization dashboard created in 'visualizations' directory")
        return "visualizations"
    
    def save_dataset(self, dataset, run_name=None):
        """Save the dataset to CSV files in a run-specific directory
        
        Args:
            dataset: Dictionary containing dataframes for users, ads, impressions, and conversions
            run_name: Name of the run (e.g., "run1", "high_bias", etc.). If None, will generate "runX" automatically
        """
        # Create base output directory if it doesn't exist
        base_dir = self.output_directory if self.output_directory else "synthetic_data"
        os.makedirs(base_dir, exist_ok=True)
        
        # If no run name provided, find the next available run number
        if run_name is None:
            existing_runs = [d for d in os.listdir(base_dir) if d.startswith("run")]
            run_numbers = [int(r.replace("run", "")) for r in existing_runs if r[3:].isdigit()]
            next_run = 1 if not run_numbers else max(run_numbers) + 1
            run_name = f"run{next_run}"
        
        # Create run-specific directory
        run_dir = os.path.join(base_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        
        # Define filenames
        users_file = os.path.join(run_dir, "users.csv")
        ads_file = os.path.join(run_dir, "ads.csv")
        impressions_file = os.path.join(run_dir, "impressions.csv")
        
        # Save files
        dataset['users'].to_csv(users_file, index=False)
        dataset['ads'].to_csv(ads_file, index=False)
        dataset['impressions'].to_csv(impressions_file, index=False)
        
        files = [users_file, ads_file, impressions_file]
        
        # Save conversions if present
        if 'conversions' in dataset and not dataset['conversions'].empty:
            conversions_file = os.path.join(run_dir, "conversions.csv")
            dataset['conversions'].to_csv(conversions_file, index=False)
            files.append(conversions_file)
            print(f"Dataset saved to '{run_dir}' directory")
        else:
            print(f"Dataset saved to '{run_dir}' directory")
            
        # Save run parameters for reproducibility
        params = {
            'num_users': self.num_users,
            'num_ads': self.num_ads,
            'bias_level': self.bias_level,
            'random_seed': self.random_seed,
            'bias_params': self.bias_params
        }
        
        with open(os.path.join(run_dir, "parameters.txt"), 'w') as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
        
        return run_dir, files

class DatasetDocumentation:
    """Documentation for the Synthetic Ad Targeting Dataset with Protected Attributes"""
    
    @staticmethod
    def get_dataset_overview():
        """Returns an overview of the dataset"""
        return """
        # Synthetic Ad Targeting Dataset with Protected Attributes
        
        This dataset simulates a digital advertising system with configurable bias parameters,
        including protected attributes for fairness analysis. It consists of four main components:
        
        1. Users: Synthetic user profiles with demographic, protected attributes, geographic, device, and behavioral attributes
        2. Ads: Synthetic advertisements with various categories and targeting parameters
        3. Impressions: Records of ads shown to users with click outcomes
        4. Conversions: Post-click actions taken by users
        
        The dataset allows researchers to study algorithmic bias in ad targeting and test
        fairness-aware algorithms for mitigating such bias, with special focus on protected 
        attributes like age, gender, ethnicity, socioeconomic status, and disability status.
        
        ## Key Features:
        
        - Configurable bias parameters for all attributes including protected classes
        - Geographic targeting patterns based on US regions and states
        - Device and behavior-based targeting
        - Temporal patterns including seasonal effects, day-of-week, and time-of-day variations
        - Ad fatigue modeling for realistic impression patterns over time
        - Social network effects showing influence of connections on ad engagement
        - Fairness metrics for evaluating and mitigating algorithmic bias
        """
    
    @staticmethod
    def get_protected_attributes():
        """Returns information about protected attributes"""
        return {
            'age_group': 'Age range (18-24, 25-34, etc.)',
            'gender': 'User gender (Male, Female, Non-binary)',
            'ethnicity': 'Ethnic background (White, Black, Hispanic, Asian, Indigenous, Other)',
            'socioeconomic_status': 'Socioeconomic status (Low, Medium, High)',
            'disability_status': 'Disability status (None, Physical, Cognitive, Sensory, Multiple)'
        }
    
    @staticmethod
    def get_fairness_metrics():
        """Returns information about fairness metrics"""
        return {
            'demographic_parity': 'Difference in outcome rates across different groups',
            'disparate_impact': 'Ratio of outcome rates between different groups',
            'equal_opportunity': 'Equal true positive rates across different groups',
            'predictive_parity': 'Equal positive predictive values across different groups'
        }
    
    @staticmethod
    def get_bias_parameters():
        """Returns information about configurable bias parameters"""
        return {
            'bias_level': 'Global bias strength (0 to 1)',
            'age_bias_strength': 'Strength of age-based targeting bias',
            'gender_bias_strength': 'Strength of gender-based targeting bias',
            'ethnicity_bias_strength': 'Strength of ethnicity-based targeting bias',
            'socioeconomic_bias_strength': 'Strength of income/education-based bias',
            'disability_bias_strength': 'Strength of disability-based targeting bias',
            'behavioral_bias_strength': 'Strength of behavior-based targeting bias',
            'geographic_bias_strength': 'Strength of location-based targeting bias',
            'device_bias_strength': 'Strength of device-based targeting bias',
            'temporal_bias_strength': 'Strength of temporal targeting patterns',
            'seasonal_bias_strength': 'Strength of seasonal targeting patterns',
            'weekday_bias_strength': 'Strength of day-of-week targeting patterns',
            'time_of_day_bias_strength': 'Strength of hour-of-day targeting patterns',
            'ad_fatigue_strength': 'Strength of ad fatigue effect over repeated exposures'
        }
    
    @staticmethod
    def get_usage_examples():
        """Returns examples of how to use the dataset generator"""
        return """
        # Basic Dataset Generation with Protected Attributes
        
        ```python
        # Create a generator with default parameters
        generator = SyntheticAdDataGenerator(
            num_users=5000,
            num_ads=1000,
            bias_level=0.7
        )
        
        # Generate dataset
        dataset = generator.generate_complete_dataset()
        
        # Analyze bias in the dataset
        bias_metrics = generator.analyze_bias(dataset)
        
        # Save the dataset to CSV files
        generator.save_dataset(dataset)
        ```
        
        # Customizing Protected Attribute Distributions
        
        ```python
        # Define custom distributions for protected attributes
        custom_distributions = {
            'gender': {'Male': 0.45, 'Female': 0.45, 'Non-binary': 0.1},
            'ethnicity': {'White': 0.5, 'Black': 0.15, 'Hispanic': 0.2, 'Asian': 0.1, 'Indigenous': 0.03, 'Other': 0.02}
        }
        
        # Create generator with custom distributions
        generator = SyntheticAdDataGenerator(
            num_users=5000,
            num_ads=1000,
            bias_level=0.7,
            custom_distributions=custom_distributions
        )
        
        # Generate dataset
        dataset = generator.generate_complete_dataset()
        ```
        
        # Comparing Different Bias Levels for Protected Attributes
        
        ```python
        # Set up bias parameters for high bias dataset
        high_bias_gen = SyntheticAdDataGenerator(
            num_users=5000,
            num_ads=1000,
            bias_level=0.9
        )
        high_bias_gen.set_bias_parameters(
            gender_bias_strength=0.9,
            ethnicity_bias_strength=0.9,
            disability_bias_strength=0.9
        )
        high_bias_dataset = high_bias_gen.generate_complete_dataset()
        
        # Set up bias parameters for low bias dataset
        low_bias_gen = SyntheticAdDataGenerator(
            num_users=5000,
            num_ads=1000,
            bias_level=0.2
        )
        low_bias_gen.set_bias_parameters(
            gender_bias_strength=0.2,
            ethnicity_bias_strength=0.2,
            disability_bias_strength=0.2
        )
        low_bias_dataset = low_bias_gen.generate_complete_dataset()
        
        # Compare fairness metrics
        high_bias_metrics = high_bias_gen.analyze_bias(high_bias_dataset)
        low_bias_metrics = low_bias_gen.analyze_bias(low_bias_dataset)
        ```
        """

def main():
    """Example usage demonstrating how to use the dataset generator with protected attributes"""
    
    print("===== TARGETED ADVERTISING BIAS SYNTHETIC DATASET GENERATOR WITH PROTECTED ATTRIBUTES =====")
    print("This tool generates synthetic data to study algorithmic bias in ad targeting with focus on protected attributes")
    
    # Create a basic generator with default parameters
    print("\n1. Generating standard dataset with medium bias level...")
    generator = SyntheticAdDataGenerator(
        num_users=100,
        num_ads=5,
        avg_impressions_per_user=20,
        bias_level=0.7,
        include_enhanced_features=True,
        output_directory="synthetic_data"
    )
    
    # Generate dataset
    dataset = generator.generate_complete_dataset()
    
    # Analyze the dataset
    print("\nAnalyzing bias patterns in the generated dataset:")
    bias_metrics = generator.analyze_bias(dataset)
    
    # Create visualizations
    generator.create_visualization_dashboard(dataset)
    
    # Save the dataset
    generator.save_dataset(dataset, run_name="baseline")
    
    # Example of generating datasets with different bias levels for comparison
    print("\n\n2. Generating dataset with high bias for protected attributes...")
    high_bias_gen = SyntheticAdDataGenerator(
        num_users=100,
        num_ads=5,
        bias_level=0.9,
        include_enhanced_features=True,
        output_directory="synthetic_data"
    )
    high_bias_gen.set_bias_parameters(
        gender_bias_strength=0.95,
        ethnicity_bias_strength=0.95,
        disability_bias_strength=0.95
    )
    high_bias_dataset = high_bias_gen.generate_complete_dataset()
    high_bias_metrics = high_bias_gen.analyze_bias(high_bias_dataset)
    high_bias_gen.save_dataset(high_bias_dataset, run_name="high_bias")
    
    print("\n\n3. Generating dataset with low bias for protected attributes...")
    low_bias_gen = SyntheticAdDataGenerator(
        num_users=100,
        num_ads=5,
        bias_level=0.2,
        include_enhanced_features=True,
        output_directory="synthetic_data"
    )
    low_bias_gen.set_bias_parameters(
        gender_bias_strength=0.2,
        ethnicity_bias_strength=0.2,
        disability_bias_strength=0.2
    )
    low_bias_dataset = low_bias_gen.generate_complete_dataset()
    low_bias_metrics = low_bias_gen.analyze_bias(low_bias_dataset)
    low_bias_gen.save_dataset(low_bias_dataset, run_name="low_bias")
    
    # Custom protected attribute distributions
    print("\n\n4. Generating dataset with custom protected attribute distributions...")
    custom_distributions = {
        'gender': {'Male': 0.33, 'Female': 0.33, 'Non-binary': 0.34},  # Equal distribution
        'ethnicity': {'White': 0.5, 'Black': 0.1, 'Hispanic': 0.2, 'Asian': 0.1, 'Indigenous': 0.05, 'Other': 0.05}
    }
    custom_dist_gen = SyntheticAdDataGenerator(
        num_users=100,
        num_ads=5,
        bias_level=0.7,
        custom_distributions=custom_distributions,
        output_directory="synthetic_data"
    )
    custom_dist_dataset = custom_dist_gen.generate_complete_dataset()
    custom_dist_metrics = custom_dist_gen.analyze_bias(custom_dist_dataset)
    custom_dist_gen.save_dataset(custom_dist_dataset, run_name="custom_dist")
    
    print("\n===== DATASET GENERATION COMPLETE =====")
    print("The datasets have been saved in run-specific directories under 'synthetic_data':")
    print("  - baseline: Standard dataset with medium bias level")
    print("  - high_bias: Dataset with high bias for protected attributes")
    print("  - low_bias: Dataset with low bias for protected attributes") 
    print("  - custom_dist: Dataset with custom protected attribute distributions")
    print("\nEach directory contains:")
    print("  - users.csv: User demographic and protected attribute data")
    print("  - ads.csv: Advertisement data")
    print("  - impressions.csv: Ad impression data with click outcomes")
    print("  - conversions.csv: Conversion data (if any)")
    print("  - parameters.txt: Generation parameters for reproducibility")
    print("\nDocumentation is available through the DatasetDocumentation class.")
    print(DatasetDocumentation.get_dataset_overview())

if __name__ == "__main__":
    main()