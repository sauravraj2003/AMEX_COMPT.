import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rankdata
import warnings
warnings.filterwarnings('ignore')

class RankingSystem:
    def __init__(self):
        self.customer_rankings = {}
        self.analysis_results = {}
        
    def calculate_map_at_k(self, df, k=7):
        """Calculate MAP@K for ranking evaluation"""
        
        def average_precision_at_k(actual, predicted, k):
            if len(predicted) > k:
                predicted = predicted[:k]
            
            score = 0.0
            num_hits = 0.0
            
            for i, p in enumerate(predicted):
                if p in actual and p not in predicted[:i]:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)
            
            return score / min(len(actual), k) if len(actual) > 0 else 0.0
        
        # Group by customer
        map_scores = []
        
        for customer_id in df['id2'].unique():
            customer_data = df[df['id2'] == customer_id].copy()
            
            # Sort by prediction probability (descending)
            customer_data = customer_data.sort_values('prediction', ascending=False)
            
            # Get actual clicks and predicted ranking
            actual_clicks = customer_data[customer_data['click'] == 1]['id3'].tolist()
            predicted_ranking = customer_data['id3'].tolist()
            
            if len(actual_clicks) > 0:
                ap = average_precision_at_k(actual_clicks, predicted_ranking, k)
                map_scores.append(ap)
        
        return np.mean(map_scores) if map_scores else 0.0
    
    def validate_predictions(self, train_data, train_predictions):
        """Validate predictions on training data"""
        
        print("=== VALIDATION ON TRAINING DATA ===")
        
        # Create validation dataframe
        val_df = train_data.copy()
        val_df['prediction'] = train_predictions
        
        # Calculate MAP@7
        map_score = self.calculate_map_at_k(val_df, k=7)
        print(f"Training MAP@7: {map_score:.6f}")
        
        # Calculate other metrics
        auc_score = roc_auc_score(train_data['click'], train_predictions)
        logloss = log_loss(train_data['click'], train_predictions)
        
        print(f"Training AUC: {auc_score:.6f}")
        print(f"Training LogLoss: {logloss:.6f}")
        
        # Click rate analysis
        click_rate = train_data['click'].mean()
        print(f"Overall Click Rate: {click_rate:.6f}")
        
        return {
            'map_at_7': map_score,
            'auc': auc_score,
            'logloss': logloss,
            'click_rate': click_rate
        }
    
    def rank_offers_for_customers(self, test_data, predictions):
        """Rank offers for each customer based on predictions"""
        
        # Add predictions to test data
        test_data_with_pred = test_data.copy()
        test_data_with_pred['prediction'] = predictions
        
        rankings = {}
        
        # For each customer, rank offers
        for customer_id in test_data_with_pred['id2'].unique():
            customer_data = test_data_with_pred[test_data_with_pred['id2'] == customer_id].copy()
            
            # Sort by prediction probability (descending)
            customer_data = customer_data.sort_values('prediction', ascending=False)
            
            # Store ranking
            rankings[customer_id] = {
                'offers': customer_data['id3'].tolist(),
                'predictions': customer_data['prediction'].tolist(),
                'unique_identifiers': customer_data['unique_identifier'].tolist()
            }
        
        self.customer_rankings = rankings
        return rankings
    
    def create_submission_file(self, test_data, predictions, team_name="team", filename=None):
        """Create submission file in required format"""
        
        if filename is None:
            filename = f'r2_submission_file_{team_name}.csv'
        
        # Create submission dataframe
        submission_df = test_data[['unique_identifier']].copy()
        submission_df['prediction'] = predictions
        
        # Sort by unique_identifier to maintain order
        submission_df = submission_df.sort_values('unique_identifier')
        
        # Ensure predictions are in valid range [0, 1]
        submission_df['prediction'] = np.clip(submission_df['prediction'], 0, 1)
        
        # Save to CSV
        submission_df.to_csv(filename, index=False)
        
        print(f"✅ Submission file saved as: {filename}")
        print(f"📊 Submission shape: {submission_df.shape}")
        print(f"📈 Prediction range: {submission_df['prediction'].min():.6f} to {submission_df['prediction'].max():.6f}")
        print(f"📊 Prediction mean: {submission_df['prediction'].mean():.6f}")
        
        return submission_df
    
    def analyze_predictions(self, test_data, predictions):
        """Analyze prediction distribution and quality"""
        
        analysis = {}
        
        # Basic statistics
        analysis['prediction_stats'] = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'q25': np.percentile(predictions, 25),
            'q50': np.percentile(predictions, 50),
            'q75': np.percentile(predictions, 75)
        }
        
        # Prediction distribution by customer
        test_data_with_pred = test_data.copy()
        test_data_with_pred['prediction'] = predictions
        
        customer_stats = test_data_with_pred.groupby('id2')['prediction'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        analysis['customer_stats'] = customer_stats
        
        # Prediction distribution by offer
        offer_stats = test_data_with_pred.groupby('id3')['prediction'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        analysis['offer_stats'] = offer_stats
        
        self.analysis_results = analysis
        return analysis
    
    def create_detailed_rankings(self, test_data, predictions, top_k=7):
        """Create detailed customer-offer rankings"""
        
        test_data_with_pred = test_data.copy()
        test_data_with_pred['prediction'] = predictions
        
        detailed_rankings = []
        
        for customer_id in test_data_with_pred['id2'].unique():
            customer_data = test_data_with_pred[test_data_with_pred['id2'] == customer_id].copy()
            
            # Sort by prediction probability (descending)
            customer_data = customer_data.sort_values('prediction', ascending=False)
            
            # Take top K offers
            top_offers = customer_data.head(top_k)
            
            for rank, (idx, row) in enumerate(top_offers.iterrows(), 1):
                detailed_rankings.append({
                    'customer_id': customer_id,
                    'offer_id': row['id3'],
                    'rank': rank,
                    'prediction': row['prediction'],
                    'unique_identifier': row['unique_identifier']
                })
        
        return pd.DataFrame(detailed_rankings)
    
    def generate_analysis_report(self, test_data, predictions, save_plots=True):
        """Generate comprehensive analysis report"""
        
        print("\n=== PREDICTION ANALYSIS REPORT ===")
        
        # Basic statistics
        print(f"\n📊 Prediction Statistics:")
        print(f"   Mean: {np.mean(predictions):.6f}")
        print(f"   Std:  {np.std(predictions):.6f}")
        print(f"   Min:  {np.min(predictions):.6f}")
        print(f"   Max:  {np.max(predictions):.6f}")
        
        # Distribution analysis
        test_data_with_pred = test_data.copy()
        test_data_with_pred['prediction'] = predictions
        
        # Customer-level analysis
        customer_stats = test_data_with_pred.groupby('id2')['prediction'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        print(f"\n👥 Customer Analysis:")
        print(f"   Total customers: {len(customer_stats)}")
        print(f"   Avg offers per customer: {customer_stats['count'].mean():.1f}")
        print(f"   Avg prediction per customer: {customer_stats['mean'].mean():.6f}")
        
        # Offer-level analysis
        offer_stats = test_data_with_pred.groupby('id3')['prediction'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        print(f"\n🎯 Offer Analysis:")
        print(f"   Total offers: {len(offer_stats)}")
        print(f"   Avg customers per offer: {offer_stats['count'].mean():.1f}")
        print(f"   Avg prediction per offer: {offer_stats['mean'].mean():.6f}")
        
        # Generate plots if requested
        if save_plots:
            self.create_analysis_plots(test_data, predictions)
        
        return {
            'prediction_stats': {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions)
            },
            'customer_stats': customer_stats,
            'offer_stats': offer_stats
        }
    
    def create_analysis_plots(self, test_data, predictions):
        """Create analysis plots"""
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Prediction distribution
        plt.subplot(2, 3, 1)
        plt.hist(predictions, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Prediction Distribution')
        plt.xlabel('Prediction Probability')
        plt.ylabel('Frequency')
        
        # Plot 2: Customer-level prediction distribution
        plt.subplot(2, 3, 2)
        test_data_with_pred = test_data.copy()
        test_data_with_pred['prediction'] = predictions
        customer_means = test_data_with_pred.groupby('id2')['prediction'].mean()
        plt.hist(customer_means, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Customer-Level Avg Predictions')
        plt.xlabel('Average Prediction')
        plt.ylabel('Number of Customers')
        
        # Plot 3: Offer-level prediction distribution
        plt.subplot(2, 3, 3)
        offer_means = test_data_with_pred.groupby('id3')['prediction'].mean()
        plt.hist(offer_means, bins=30, alpha=0.7, edgecolor='black')
        plt.title('Offer-Level Avg Predictions')
        plt.xlabel('Average Prediction')
        plt.ylabel('Number of Offers')
        
        # Plot 4: Top customers by prediction
        plt.subplot(2, 3, 4)
        top_customers = customer_means.sort_values(ascending=False).head(20)
        plt.bar(range(len(top_customers)), top_customers.values)
        plt.title('Top 20 Customers by Avg Prediction')
        plt.xlabel('Customer Rank')
        plt.ylabel('Average Prediction')
        
        # Plot 5: Top offers by prediction
        plt.subplot(2, 3, 5)
        top_offers = offer_means.sort_values(ascending=False).head(20)
        plt.bar(range(len(top_offers)), top_offers.values)
        plt.title('Top 20 Offers by Avg Prediction')
        plt.xlabel('Offer Rank')
        plt.ylabel('Average Prediction')
        
        # Plot 6: Prediction vs rank correlation
        plt.subplot(2, 3, 6)
        ranks = rankdata(-predictions)
        plt.scatter(ranks[:1000], predictions[:1000], alpha=0.5)
        plt.title('Prediction vs Rank (Sample)')
        plt.xlabel('Rank')
        plt.ylabel('Prediction')
        
        plt.tight_layout()
        plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Analysis plots saved as 'prediction_analysis.png'")


def post_process_predictions(predictions, method='calibration'):
    """Post-process predictions for better ranking"""
    
    if method == 'calibration':
        # Sigmoid calibration to ensure [0,1] range
        predictions = np.clip(predictions, -10, 10)  # Prevent overflow
        predictions = 1 / (1 + np.exp(-predictions))
    
    elif method == 'rank_normalization':
        # Rank-based normalization
        predictions = rankdata(predictions) / len(predictions)
    
    elif method == 'min_max_scaling':
        # Min-max scaling
        pred_min, pred_max = np.min(predictions), np.max(predictions)
        if pred_max > pred_min:
            predictions = (predictions - pred_min) / (pred_max - pred_min)
        else:
            predictions = np.ones_like(predictions) * 0.5
    
    elif method == 'quantile_normalization':
        # Quantile-based normalization
        predictions = rankdata(predictions, method='average') / len(predictions)
    
    return predictions


def generate_final_submission(train_data, test_data, add_trans, add_event, offer_metadata, team_name="team"):
    """Complete pipeline to generate final submission"""
    
    print("=== AMEX OFFERINGS PERSONALIZATION SOLUTION ===")
    print(f"Team: {team_name}")
    print(f"Data shapes - Train: {train_data.shape}, Test: {test_data.shape}")
    
    # Step 1: Feature Engineering
    print("\n1. 🔧 Feature Engineering...")
    try:
        from part2_feature_engineering import prepare_features
        train_processed, test_processed, feature_engineer = prepare_features(
            train_data, test_data, add_trans, add_event, offer_metadata
        )
        print(f"   ✅ Features prepared. Train: {train_processed.shape}, Test: {test_processed.shape}")
    except ImportError:
        print("   ⚠️  Feature engineering module not found. Using basic features.")
        # Basic feature preparation
        train_processed = train_data.copy()
        test_processed = test_data.copy()
    
    # Step 2: Model Training
    print("\n2. 🤖 Model Training...")
    try:
        from part3_model_building import train_and_predict
        predictions, ensemble, feature_importance = train_and_predict(
            train_processed, test_processed
        )
        print(f"   ✅ Model trained and predictions generated.")
    except ImportError:
        print("   ⚠️  Model building module not found. Using dummy predictions.")
        # Dummy predictions for testing
        predictions = np.random.random(len(test_processed))
    
    # Step 3: Post-processing
    print("\n3. 🔄 Post-processing predictions...")
    predictions = post_process_predictions(predictions, method='calibration')
    print(f"   ✅ Predictions post-processed. Range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}]")
    
    # Step 4: Ranking and Analysis
    print("\n4. 📊 Creating rankings and analysis...")
    ranking_system = RankingSystem()
    
    # Validate on training data if available
    if 'click' in train_processed.columns:
        try:
            train_predictions = ensemble.predict_proba(train_processed.drop(['click'] + [col for col in train_processed.columns if col.startswith('id')], axis=1))[:, 1]
            train_predictions = post_process_predictions(train_predictions, method='calibration')
            validation_results = ranking_system.validate_predictions(train_processed, train_predictions)
        except:
            print("   ⚠️  Validation skipped due to missing components.")
    
    # Create rankings
    rankings = ranking_system.rank_offers_for_customers(test_processed, predictions)
    print(f"   ✅ Rankings created for {len(rankings)} customers.")
    
    # Generate analysis
    analysis = ranking_system.generate_analysis_report(test_processed, predictions)
    
    # Step 5: Create submission file
    print("\n5. 📝 Creating submission file...")
    submission_df = ranking_system.create_submission_file(test_processed, predictions, team_name)
    
    # Step 6: Generate detailed rankings
    print("\n6. 📋 Generating detailed rankings...")
    detailed_rankings = ranking_system.create_detailed_rankings(test_processed, predictions, top_k=7)
    detailed_rankings.to_csv(f'detailed_rankings_{team_name}.csv', index=False)
    print(f"   ✅ Detailed rankings saved as 'detailed_rankings_{team_name}.csv'")
    
    # Final summary
    print("\n=== SUBMISSION SUMMARY ===")
    print(f"✅ Submission file: r2_submission_file_{team_name}.csv")
    print(f"✅ Detailed rankings: detailed_rankings_{team_name}.csv")
    print(f"✅ Analysis plots: prediction_analysis.png")
    print(f"📊 Total predictions: {len(predictions)}")
    print(f"📊 Unique customers: {test_processed['id2'].nunique()}")
    print(f"📊 Unique offers: {test_processed['id3'].nunique()}")
    
    return {
        'submission_df': submission_df,
        'detailed_rankings': detailed_rankings,
        'analysis': analysis,
        'ranking_system': ranking_system,
        'predictions': predictions
    }


# Example usage function
def run_complete_pipeline():
    """Run the complete pipeline with sample data"""
    
    # Load your data here
    # train_data = pd.read_parquet('train_data.parquet')
    # test_data = pd.read_parquet('test_data.parquet')
    # add_trans = pd.read_parquet('add_trans.parquet')
    # add_event = pd.read_parquet('add_event.parquet')
    # offer_metadata = pd.read_parquet('offer_metadata.parquet')
    
    # For demonstration, create dummy data
    print("⚠️  Using dummy data for demonstration. Replace with actual data loading.")
    
    # Replace this with actual data loading
    train_data = pd.DataFrame({
        'unique_identifier': range(1000),
        'id2': np.random.randint(1, 101, 1000),
        'id3': np.random.randint(1, 21, 1000),
        'click': np.random.binomial(1, 0.1, 1000),
        'feature1': np.random.random(1000),
        'feature2': np.random.random(1000)
    })
    
    test_data = pd.DataFrame({
        'unique_identifier': range(1000, 1500),
        'id2': np.random.randint(1, 101, 500),
        'id3': np.random.randint(1, 21, 500),
        'feature1': np.random.random(500),
        'feature2': np.random.random(500)
    })
    
    # Empty additional datasets for demo
    add_trans = pd.DataFrame()
    add_event = pd.DataFrame()
    offer_metadata = pd.DataFrame()
    
    # Run the pipeline
    results = generate_final_submission(
        train_data, test_data, add_trans, add_event, offer_metadata, 
        team_name="your_team_name"
    )
    
    return results

if __name__ == "__main__":
    # Run the complete pipeline
    results = run_complete_pipeline()