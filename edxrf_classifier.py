"""
EDXRF Classification Model - Inference Module
"""
import joblib
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EDXRFClassifier:
    def __init__(self, model_path, scaler_path, metadata_path):
        try:
            logger.info("Loading Random Forest model...")
            self.model = joblib.load(model_path)
            logger.info("Loading feature scaler...")
            self.scaler = joblib.load(scaler_path)
            logger.info("Loading model metadata...")
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.feature_names = self.metadata['features']
            self.classes = self.metadata['classes']
            self.loocv_accuracy = self.metadata['loocv_accuracy']
            logger.info("Model successfully initialized")
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def validate_input(self, elemental_dict):
        missing_features = []
        for feat in self.feature_names:
            if feat not in elemental_dict:
                missing_features.append(feat)
        
        if missing_features:
            return False, f"Missing features: {missing_features}"
        
        for feat in self.feature_names:
            try:
                float(elemental_dict[feat])
            except (ValueError, TypeError):
                return False, f"Non-numeric value for {feat}"
        
        return True, None
    
    def predict_single(self, elemental_dict):
        try:
            is_valid, error_msg = self.validate_input(elemental_dict)
            if not is_valid:
                raise ValueError(error_msg)
            
            X = pd.DataFrame([elemental_dict])
            X = X[self.feature_names]
            X_scaled = self.scaler.transform(X)
            
            prediction = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = max(probabilities)
            
            result = {
                'predicted_class': int(prediction),
                'confidence': float(round(confidence, 4)),
                'class_0_probability': float(round(probabilities[0], 4)),
                'class_1_probability': float(round(probabilities[1], 4)),
                'timestamp': datetime.now().isoformat(),
                'model_accuracy': self.loocv_accuracy,
                'warning': None
            }
            
            if confidence < 0.7:
                result['warning'] = "Low confidence prediction"
            
            return result
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_batch(self, data_list):
        results = []
        errors = []
        
        for i, sample in enumerate(data_list):
            try:
                result = self.predict_single(sample)
                results.append({'sample_index': i, 'result': result})
            except Exception as e:
                errors.append({'sample_index': i, 'error': str(e)})
        
        return {
            'successful': results,
            'failed': errors,
            'total': len(data_list),
            'success_rate': len(results) / len(data_list) if data_list else 0
        }
    
    def get_model_info(self):
        return {
            'model_type': self.metadata['model_type'],
            'loocv_accuracy': f"{self.metadata['loocv_accuracy']*100:.1f}%",
            'training_samples': self.metadata['training_samples'],
            'classes': self.classes,
            'features': self.feature_names,
            'status': 'Production Ready'
        }
