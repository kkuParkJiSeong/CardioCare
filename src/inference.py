import os
import sys
import argparse
import pickle
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="CardioCare Inference Script")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True, help="Path to save predictions CSV file")
    args = parser.parse_args()

    # Load final model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models/final_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.", file=sys.stderr)
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Read input CSV
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}", file=sys.stderr)
        sys.exit(1)
        
    df = pd.read_csv(args.input)
    
    # Check if 'target' column is present and drop it for prediction if it is
    feature_df = df.copy()
    if 'target' in feature_df.columns:
        feature_df = feature_df.drop(columns=['target'])
        
    # Check shape/features
    expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    # If the input doesn't have matching headers but has 13 columns, we can assign them
    if list(feature_df.columns) != expected_features:
        if feature_df.shape[1] == 13:
            feature_df.columns = expected_features
        else:
            print(f"Error: Expected 13 columns, but got {feature_df.shape[1]} columns.", file=sys.stderr)
            sys.exit(1)
            
    # Perform prediction
    predictions = model.predict(feature_df)
    probabilities = model.predict_proba(feature_df)[:, 1] if hasattr(model, "predict_proba") else [None] * len(predictions)
    
    # Save predictions
    output_df = df.copy()
    output_df['prediction'] = predictions
    output_df['probability'] = probabilities
    
    # Ensure directory of output exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    output_df.to_csv(args.output, index=False)
    print(f"Predictions saved successfully to {args.output}")

if __name__ == "__main__":
    main()
