import os
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from preprocessing import preprocessing_data

logger = logging.getLogger("CardioCare_Monitor")
logger.setLevel(logging.INFO)

log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.log")
file_handler = logging.FileHandler(log_file, encoding="utf-8")
formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def instrumented_inference(df, model, true_labels=None, model_version="1.0"):
    """
    추론 경로에 logging 기반 계측 추가:
    타임스탬프, 모델 버전, 입력 shape, 예측값, (가능한 경우) 실제 정답을 파일로 기록합니다.
    """
    predictions = model.predict(df)
    
    log_msg = (
        f"Model Version: {model_version} | "
        f"Input Shape: {df.shape} | "
        f"Predictions (first 10): {predictions[:10].tolist()} | "
        f"Prediction Count: {len(predictions)}"
    )
    if true_labels is not None:
        log_msg += f" | True Labels (first 10): {true_labels[:10].tolist()}"
        
    logger.info(log_msg)
    return predictions

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "../data/processed.cleveland.data")
    model_path = os.path.join(current_dir, "models/final_model.pkl")
    
    print("--- 1. Loading Data and Model ---")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        return
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        
    df = pd.read_csv(data_path, header=None, na_values='?')
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df['target'] = (df['target'] > 0).astype(int)
    
    df_preprocessed = preprocessing_data(df)
    
    X = df_preprocessed.drop(columns=['target'])
    y = df_preprocessed['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Loaded train features shape: {X_train.shape}")
    print(f"Loaded test features shape: {X_test.shape}")
    
    print("\n--- 2. Running Instrumented Inference ---")
    test_preds = instrumented_inference(X_test, model, true_labels=y_test.values)
    print(f"Inference completed and logged to {log_file}")
    
    print("\n--- 3. Simulating Feature Drift ---")
    # chol (cholesterol) 특성에 대해 평균을 +30 이동하고 분산을 증가시킵니다.
    # 원래 chol 분포
    orig_chol_mean = X_test['chol'].mean()
    orig_chol_std = X_test['chol'].std()
    print(f"Original 'chol' - Mean: {orig_chol_mean:.2f}, Std: {orig_chol_std:.2f}")
    
    X_test_drifted = X_test.copy()
    # 분산을 증가시키기 위해 평균과의 편차를 1.5배 키우고, 평균 자체를 +30 이동시킵니다.
    X_test_drifted['chol'] = (X_test_drifted['chol'] - orig_chol_mean) * 1.5 + orig_chol_mean + 30.0
    
    drifted_chol_mean = X_test_drifted['chol'].mean()
    drifted_chol_std = X_test_drifted['chol'].std()
    print(f"Drifted 'chol' - Mean: {drifted_chol_mean:.2f}, Std: {drifted_chol_std:.2f}")
    
    print("\n--- 4. Kolmogorov-Smirnov Test for Continuous Features ---")
    continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    drifted_features = []
    
    for col in continuous_features:
        stat, p_val = ks_2samp(X_train[col], X_test_drifted[col])
        is_drifted = p_val < 0.05
        print(f"Feature '{col:8s}': KS stat = {stat:.4f}, p-value = {p_val:.4e} {'[DRIFT DETECTED!]' if is_drifted else '[Normal]'}")
        if is_drifted:
            drifted_features.append(col)
            
    print(f"Flagged drifted features (p < 0.05): {drifted_features}")
    
    print("\n--- 5. Performance Comparison (Original vs Drifted) ---")
    orig_bal_acc = balanced_accuracy_score(y_test, test_preds)
    
    drifted_preds = model.predict(X_test_drifted)
    drifted_bal_acc = balanced_accuracy_score(y_test, drifted_preds)
    
    print(f"Original Test Set Balanced Accuracy: {orig_bal_acc:.4f}")
    print(f"Drifted Test Set Balanced Accuracy:  {drifted_bal_acc:.4f}")
    print(f"Performance Drop: {orig_bal_acc - drifted_bal_acc:.4f}")
    
    instrumented_inference(X_test_drifted, model, true_labels=y_test.values)
    
    print("\n--- 6. Generating Drift Metrics Time-series Plot ---")
    # 10일간 서서히 chol 수치가 증가(평균 +0에서 +30까지 점진적으로 증가)하는 시나리오를 시뮬레이션합니다.
    days = list(range(1, 11))
    daily_bal_accs = []
    daily_p_values = []
    
    for day in days:
        # 점진적 shift 가중치 (0%에서 100%까지)
        weight = (day - 1) / 9.0
        shift_amount = weight * 30.0
        var_factor = 1.0 + weight * 0.5 # 분산 1.0배에서 1.5배로 점진적 증가
        
        X_daily = X_test.copy()
        X_daily['chol'] = (X_daily['chol'] - orig_chol_mean) * var_factor + orig_chol_mean + shift_amount
        
        preds = model.predict(X_daily)
        bal_acc = balanced_accuracy_score(y_test, preds)
        daily_bal_accs.append(bal_acc)
        
        _, p_val = ks_2samp(X_train['chol'], X_daily['chol'])
        daily_p_values.append(p_val)
        
    # 플롯 생성
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    color = 'tab:blue'
    ax1.set_xlabel('Simulated Day')
    ax1.set_ylabel('Balanced Accuracy', color=color)
    ax1.plot(days, daily_bal_accs, marker='o', color=color, linewidth=2, label='Balanced Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('KS Test p-value (chol)', color=color)
    ax2.plot(days, daily_p_values, marker='s', linestyle='--', color=color, linewidth=2, label='KS p-value')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    ax2.axhline(y=0.05, color='gray', linestyle=':', label='Drift Threshold (0.05)')
    
    plt.title('Performance & Feature Drift (chol) Time-series Tracking')
    
    plot_path = os.path.join(current_dir, "drift_metrics_timeseries.png")
    fig.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved time-series plot to {plot_path}")

if __name__ == "__main__":
    main()
