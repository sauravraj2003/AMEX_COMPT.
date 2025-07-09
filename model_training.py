import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

class ModelEnsemble:
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.feature_importance = {}
        
    def create_models(self):
        """Create different models for ensemble"""
        
        models = {
            'lgb': lgb.LGBMClassifier(
                objective='binary',
                boosting_type='gbdt',
                num_leaves=31,
                learning_rate=0.05,
                feature_fraction=0.9,
                bagging_fraction=0.8,
                bagging_freq=5,
                verbose=-1,
                random_state=42
            ),
            
            'xgb': xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                max_depth=6,
                learning_rate=0.05,
                n_estimators=1000,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            
            'rf': RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            
            'gb': GradientBoostingClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            
            'lr': LogisticRegression(
                penalty='l2',
                C=1.0,
                random_state=42,
                max_iter=1000
            )
        }
        
        return models
    
    def train_single_model(self, model, X_train, y_train, X_val, y_val, model_name):
        """Train a single model"""
        print(f"Training {model_name}...")
        
        if model_name in ['lgb', 'xgb']:
            # Early stopping for gradient boosting
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict_proba(X_train)[:, 1]
        val_pred = model.predict_proba(X_val)[:, 1]
        
        # Metrics
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        
        train_ap = average_precision_score(y_train, train_pred)
        val_ap = average_precision_score(y_val, val_pred)
        
        print(f"{model_name} - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}")
        print(f"{model_name} - Train AP: {train_ap:.4f}, Val AP: {val_ap:.4f}")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_name] = model.feature_importances_
        
        return model, val_pred, val_auc
    
    def train_ensemble(self, X_train, y_train, feature_names, n_folds=5):
        """Train ensemble of models with cross-validation"""
        print("Training ensemble models...")
        
        models = self.create_models()
        
        # Cross-validation setup
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Store out-of-fold predictions
        oof_predictions = np.zeros((len(X_train), len(models)))
        model_scores = {}
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            print(f"\nFold {fold + 1}/{n_folds}")
            
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y_train.iloc[val_idx]
            
            for i, (model_name, model) in enumerate(models.items()):
                # Train model
                trained_model, val_pred, val_score = self.train_single_model(
                    model, X_fold_train, y_fold_train, X_fold_val, y_fold_val, model_name
                )
                
                # Store predictions
                oof_predictions[val_idx, i] = val_pred
                
                # Store model for first fold
                if fold == 0:
                    self.models[model_name] = trained_model
                    model_scores[model_name] = []
                
                model_scores[model_name].append(val_score)
        
        # Calculate average scores
        for model_name in models.keys():
            avg_score = np.mean(model_scores[model_name])
            print(f"\n{model_name} - Average CV AUC: {avg_score:.4f}")
            self.model_weights[model_name] = avg_score
        
        return oof_predictions
    
    def calculate_map_at_k(self, y_true, y_pred, k=7):
        """Calculate MAP@K metric"""
        def apk(actual, predicted, k):
            if len(predicted) > k:
                predicted = predicted[:k]
            
            score = 0.0
            num_hits = 0.0
            
            for i, p in enumerate(predicted):
                if p in actual and p not in predicted[:i]:
                    num_hits += 1.0
                    score += num_hits / (i + 1.0)
            
            return score / min(len(actual), k)
        
        # Group by customer
        df = pd.DataFrame({'customer': range(len(y_true)), 'actual': y_true, 'pred': y_pred})
        
        # For each customer, rank offers by prediction
        map_scores = []
        for customer in df['customer'].unique():
            customer_data = df[df['customer'] == customer]
            
            # Sort by prediction descending
            customer_data = customer_data.sort_values('pred', ascending=False)
            
            actual_clicks = customer_data[customer_data['actual'] == 1].index.tolist()
            predicted_order = customer_data.index.tolist()
            
            if len(actual_clicks) > 0:
                ap = apk(actual_clicks, predicted_order, k)
                map_scores.append(ap)
        
        return np.mean(map_scores) if map_scores else 0.0
    
    def predict_ensemble(self, X_test, feature_names):
        """Make predictions using ensemble"""
        print("Making ensemble predictions...")
        
        predictions = np.zeros((len(X_test), len(self.models)))
        
        for i, (model_name, model) in enumerate(self.models.items()):
            pred = model.predict_proba(X_test)[:, 1]
            predictions[:, i] = pred
        
        # Weighted average based on validation performance
        weights = np.array([self.model_weights[name] for name in self.models.keys()])
        weights = weights / weights.sum()  # Normalize
        
        ensemble_pred = np.average(predictions, axis=1, weights=weights)
        
        return ensemble_pred
    
    def get_feature_importance(self, feature_names):
        """Get aggregated feature importance"""
        if not self.feature_importance:
            return None
        
        # Average feature importance across models
        avg_importance = np.zeros(len(feature_names))
        
        for model_name, importance in self.feature_importance.items():
            if len(importance) == len(feature_names):
                avg_importance += importance
        
        avg_importance = avg_importance / len(self.feature_importance)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

# Usage example
def train_and_predict(train_data, test_data, target_col='click'):
    """Main training and prediction pipeline"""
    
    # Prepare features
    feature_cols = [col for col in train_data.columns if col not in ['click', 'id2', 'id3', 'unique_identifier']]
    
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    X_test = test_data[feature_cols]
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Target distribution: {y_train.value_counts()}")
    
    # Initialize ensemble
    ensemble = ModelEnsemble()
    
    # Train models
    oof_predictions = ensemble.train_ensemble(X_train, y_train, feature_cols)
    
    # Make predictions
    test_predictions = ensemble.predict_ensemble(X_test, feature_cols)
    
    # Feature importance
    feature_importance = ensemble.get_feature_importance(feature_cols)
    
    return test_predictions, ensemble, feature_importance