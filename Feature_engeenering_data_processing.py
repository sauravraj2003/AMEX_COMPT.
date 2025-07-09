import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        
    def create_transaction_features(self, main_data, transaction_data):
        """Create features from transaction data"""
        print("Creating transaction features...")
        
        # Aggregate transaction features by customer
        trans_agg = transaction_data.groupby('id2').agg({
            'f367': ['sum', 'mean', 'count', 'std', 'min', 'max'],  # Transaction Amount
            'f368': ['nunique'],  # Product ID
            'f369': ['mean'],  # Debit/Credit ratio
        }).reset_index()
        
        # Flatten column names
        trans_agg.columns = ['id2'] + [f'trans_{col[0]}_{col[1]}' for col in trans_agg.columns[1:]]
        
        # Recent transaction features (if date available)
        if 'f370' in transaction_data.columns:
            transaction_data['f370'] = pd.to_datetime(transaction_data['f370'], errors='coerce')
            recent_trans = transaction_data[transaction_data['f370'] >= transaction_data['f370'].max() - pd.Timedelta(days=30)]
            
            recent_agg = recent_trans.groupby('id2').agg({
                'f367': ['sum', 'mean', 'count'],
                'f368': ['nunique']
            }).reset_index()
            
            recent_agg.columns = ['id2'] + [f'recent_trans_{col[0]}_{col[1]}' for col in recent_agg.columns[1:]]
            trans_agg = trans_agg.merge(recent_agg, on='id2', how='left')
        
        # Merge with main data
        main_data = main_data.merge(trans_agg, on='id2', how='left')
        
        return main_data
    
    def create_event_features(self, main_data, event_data):
        """Create features from event data"""
        print("Creating event features...")
        
        # Customer-Offer interaction features
        event_agg = event_data.groupby(['id2', 'id3']).agg({
            'id4': ['count'],  # Impression count
            'id7': ['count'],  # Click count
        }).reset_index()
        
        event_agg.columns = ['id2', 'id3', 'impression_count', 'click_count']
        event_agg['historical_ctr'] = event_agg['click_count'] / (event_agg['impression_count'] + 1)
        
        # Customer level aggregations
        customer_agg = event_data.groupby('id2').agg({
            'id3': ['nunique'],  # Unique offers seen
            'id4': ['count'],  # Total impressions
            'id7': ['count'],  # Total clicks
            'f374': ['nunique'],  # Unique industries
        }).reset_index()
        
        customer_agg.columns = ['id2'] + [f'customer_{col[0]}_{col[1]}' for col in customer_agg.columns[1:]]
        customer_agg['customer_overall_ctr'] = customer_agg['customer_id7_count'] / (customer_agg['customer_id4_count'] + 1)
        
        # Merge features
        main_data = main_data.merge(customer_agg, on='id2', how='left')
        main_data = main_data.merge(event_agg, on=['id2', 'id3'], how='left')
        
        return main_data
    
    def create_offer_features(self, main_data, offer_data):
        """Create features from offer metadata"""
        print("Creating offer features...")
        
        # Merge offer metadata
        main_data = main_data.merge(offer_data, on='id3', how='left')
        
        # Encode categorical variables
        categorical_cols = ['id9', 'id10', 'id11', 'f374']  # Offer name, industry, brand, etc.
        
        for col in categorical_cols:
            if col in main_data.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    main_data[f'{col}_encoded'] = self.encoders[col].fit_transform(main_data[col].astype(str))
                else:
                    main_data[f'{col}_encoded'] = self.encoders[col].transform(main_data[col].astype(str))
        
        return main_data
    
    def create_time_features(self, main_data):
        """Create time-based features"""
        print("Creating time features...")
        
        # If timestamp columns exist
        time_cols = ['id4', 'id7', 'id12', 'id13']  # impression, click, start, end timestamps
        
        for col in time_cols:
            if col in main_data.columns:
                main_data[col] = pd.to_datetime(main_data[col], errors='coerce')
                main_data[f'{col}_hour'] = main_data[col].dt.hour
                main_data[f'{col}_day_of_week'] = main_data[col].dt.dayofweek
                main_data[f'{col}_month'] = main_data[col].dt.month
        
        return main_data
    
    def handle_missing_values(self, data):
        """Handle missing values"""
        print("Handling missing values...")
        
        # Fill numerical columns with median
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
        
        # Fill categorical columns with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mode()[0] if len(data[col].mode()) > 0 else 'unknown')
        
        return data
    
    def scale_features(self, train_data, test_data, feature_cols):
        """Scale numerical features"""
        print("Scaling features...")
        
        scaler = StandardScaler()
        train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
        test_data[feature_cols] = scaler.transform(test_data[feature_cols])
        
        self.scalers['main'] = scaler
        
        return train_data, test_data
    
    def select_features(self, X, y, k=100):
        """Select top k features"""
        print(f"Selecting top {k} features...")
        
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        self.feature_selector = selector
        
        print(f"Selected features: {len(selected_features)}")
        return X_selected, selected_features

# Usage example
def prepare_features(train_data, test_data, add_trans, add_event, offer_metadata):
    """Main feature preparation pipeline"""
    
    fe = FeatureEngineer()
    
    # Combine train and test for consistent feature engineering
    train_data['is_train'] = 1
    test_data['is_train'] = 0
    
    # Add dummy target for test data
    if 'click' not in test_data.columns:
        test_data['click'] = 0
    
    combined_data = pd.concat([train_data, test_data], ignore_index=True)
    
    # Create features
    combined_data = fe.create_transaction_features(combined_data, add_trans)
    combined_data = fe.create_event_features(combined_data, add_event)
    combined_data = fe.create_offer_features(combined_data, offer_metadata)
    combined_data = fe.create_time_features(combined_data)
    combined_data = fe.handle_missing_values(combined_data)
    
    # Split back
    train_processed = combined_data[combined_data['is_train'] == 1].drop(['is_train'], axis=1)
    test_processed = combined_data[combined_data['is_train'] == 0].drop(['is_train'], axis=1)
    
    return train_processed, test_processed, fe