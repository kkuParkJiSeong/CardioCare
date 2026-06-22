import os
import sys
import argparse
import pickle
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="CardioCare Inference Script")
    parser.add_argument("--input", type=str, default=None, help="Path to input CSV file")
    parser.add_argument("--output", type=str, default=None, help="Path to save predictions CSV file")
    args = parser.parse_args()

    # Load final model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(current_dir, "models/final_model.pkl")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.", file=sys.stderr)
        sys.exit(1)
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    # Determine input source
    if args.input is not None:
        input_path = args.input
        if not os.path.exists(input_path):
            print(f"Error: Input file not found at {input_path}", file=sys.stderr)
            sys.exit(1)
        df = pd.read_csv(input_path)
    else:
        # Fallback to a default sample within the project
        sample_input_path = os.path.join(project_root, "sample_input.csv")
        if os.path.exists(sample_input_path):
            print(f"No input file specified. Using default sample input: {sample_input_path}")
            df = pd.read_csv(sample_input_path)
        else:
            # Fallback to reading first 5 lines of processed.cleveland.data if sample_input.csv is missing
            cleveland_data_path = os.path.join(project_root, "data/processed.cleveland.data")
            if os.path.exists(cleveland_data_path):
                print(f"No input file specified. Generating a default sample from {cleveland_data_path}...")
                raw_df = pd.read_csv(cleveland_data_path, header=None, nrows=5)
                expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                raw_df.columns = expected_features + ['target']
                df = raw_df
            else:
                print("Error: No input specified and default data files could not be found.", file=sys.stderr)
                sys.exit(1)
    
    # Check if 'target' column is present and drop it for prediction if it is
    feature_df = df.copy()
    if 'target' in feature_df.columns:
        feature_df = feature_df.drop(columns=['target'])
        
    # Check shape/features
    expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    if list(feature_df.columns) != expected_features:
        if feature_df.shape[1] == 13:
            feature_df.columns = expected_features
        else:
            print(f"Error: Expected 13 columns, but got {feature_df.shape[1]} columns.", file=sys.stderr)
            sys.exit(1)
            
    # Perform prediction
    predictions = model.predict(feature_df)
    probabilities = model.predict_proba(feature_df)[:, 1] if hasattr(model, "predict_proba") else [None] * len(predictions)
    
    # Create output dataframe
    output_df = df.copy()
    output_df['prediction'] = predictions
    output_df['probability'] = probabilities
    
    # Save predictions or display them
    if args.output is not None:
        output_dir = os.path.dirname(os.path.abspath(args.output))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_df.to_csv(args.output, index=False)
        print(f"Predictions saved successfully to {args.output}")
    else:
        print("\n================== CARDIO CARE INFERENCE RESULTS ==================")
        # Pretty printing select columns for clarity
        display_cols = ['age', 'sex', 'cp', 'chol', 'prediction', 'probability']
        # Filter display cols to what is available
        actual_display_cols = [c for c in display_cols if c in output_df.columns]
        print(output_df[actual_display_cols].to_string(index=False))
        print("===================================================================\n")

if __name__ == "__main__":
    main()
