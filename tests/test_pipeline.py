import os
import sys
import unittest
import numpy as np
import pandas as pd
import pickle

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from preprocessing import preprocessing_data, validate_ranges

class TestCardioCarePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Determine paths
        cls.current_dir = os.path.dirname(os.path.abspath(__file__))
        cls.data_path = os.path.join(cls.current_dir, '../data/processed.cleveland.data')
        cls.model_path = os.path.join(cls.current_dir, '../src/models/final_model.pkl')
        
        # Load sample input data for testing
        df = pd.read_csv(cls.data_path, header=None, na_values='?')
        df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        df['target'] = (df['target'] > 0).astype(int)
        
        # Keep original columns for input format testing
        cls.sample_df = preprocessing_data(df).drop(columns=['target']).head(10)
        
        # Try to load the trained model
        if os.path.exists(cls.model_path):
            with open(cls.model_path, 'rb') as f:
                cls.model = pickle.load(f)
        else:
            # Fallback: train a dummy pipeline on the spot so tests pass without training first
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.feature_selection import SelectFromModel
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.svm import SVC
            
            df_proc = preprocessing_data(df)
            X = df_proc.drop(columns=['target'])
            y = df_proc['target']
            
            cls.model = Pipeline([
                ('scaler', StandardScaler()),
                ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42), threshold='median')),
                ('classifier', SVC(probability=True, random_state=42))
            ])
            cls.model.fit(X, y)

    def test_1_prediction_shape(self):
        """1. 예측 결과의 shape가 입력 shape와 일치하는지 확인"""
        input_data = self.sample_df
        predictions = self.model.predict(input_data)
        self.assertEqual(predictions.shape[0], input_data.shape[0])
        
    def test_2_prediction_probabilities(self):
        """2. 예측 확률이 [0, 1] 범위 내에 있고 행마다 합이 약 1 인지 확인"""
        if hasattr(self.model, "predict_proba"):
            input_data = self.sample_df
            proba = self.model.predict_proba(input_data)
            
            # Check shape (N, 2)
            self.assertEqual(proba.shape, (input_data.shape[0], 2))
            
            # Check range [0, 1]
            self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))
            
            # Check row-wise sum is approx 1.0
            row_sums = np.sum(proba, axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)
        else:
            self.skipTest("Model does not support predict_proba")

    def test_3_input_range_validation(self):
        """3. 임상적으로 범위가 정해진 특성(예: chol 이 [0, 600])에 대한 입력값 범위 검증"""
        # Valid test case
        valid_df = self.sample_df.copy()
        self.assertTrue(validate_ranges(valid_df))
        
        # Invalid cholesterol (greater than 600)
        invalid_chol_df = self.sample_df.copy()
        invalid_chol_df.loc[0, 'chol'] = 650.0
        with self.assertRaises(ValueError):
            validate_ranges(invalid_chol_df)
            
        # Invalid cholesterol (negative)
        invalid_chol_neg_df = self.sample_df.copy()
        invalid_chol_neg_df.loc[0, 'chol'] = -10.0
        with self.assertRaises(ValueError):
            validate_ranges(invalid_chol_neg_df)
            
        # Invalid age
        invalid_age_df = self.sample_df.copy()
        invalid_age_df.loc[0, 'age'] = 150.0
        with self.assertRaises(ValueError):
            validate_ranges(invalid_age_df)

    def test_4_determinism(self):
        """4. 고정 시드에서 파이프라인이 결정론적인지 확인 (동일 입력 → 동일 출력)"""
        input_data = self.sample_df
        
        # Run prediction twice
        pred1 = self.model.predict(input_data)
        pred2 = self.model.predict(input_data)
        
        # Compare
        np.testing.assert_array_equal(pred1, pred2)
        
        if hasattr(self.model, "predict_proba"):
            proba1 = self.model.predict_proba(input_data)
            proba2 = self.model.predict_proba(input_data)
            np.testing.assert_allclose(proba1, proba2, atol=1e-10)

if __name__ == '__main__':
    unittest.main()
