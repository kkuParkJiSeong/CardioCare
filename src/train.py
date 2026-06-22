import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

from preprocessing import preprocessing_data

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '../data/processed.cleveland.data')
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, header=None, na_values='?')
    df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df['target'] = (df['target'] > 0).astype(int)
    
    # Preprocess
    df = preprocessing_data(df)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 평가 데이터의 통계적 안정성을 유지하기 위해 8:2 분할 비율을 선택했습니다.
    # 시드는 42로 고정
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data Split: Train shape={X_train.shape}, Test shape={X_test.shape}")
    
    # Disable autologging to prevent MLflow from automatically creating duplicate runs and charts
    mlflow.autolog(disable=True)
    
    # Set MLflow experiment
    experiment_name = "CardioCare_Heart_Disease_Prediction"
    client = mlflow.MlflowClient()
    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None and exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(experiment_name)
    
    # Models to compare
    models_config = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "family": "logistic_regression",
            "params": {"C": 1.0}
        },
        "SVC": {
            "model": CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
            "family": "svc",
            "params": {"C": 1.0, "kernel": "rbf"}
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "family": "random_forest",
            "params": {"n_estimators": 100, "max_depth": None}
        }
    }
    
    results = {}
    
    # 3. Model Training & Evaluation
    for model_name, config in models_config.items():
        print(f"\n--- Training {model_name} ---")
        
        # Define Pipeline
        # 누수 방지를 위해 StandardScaler와 SelectFromModel을 Pipeline으로 구성합니다.
        # 이로 인해 CV 시 각 fold의 validation set에 데이터 누수가 발생하지 않습니다.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectFromModel(RandomForestClassifier(random_state=42), threshold='median')),
            ('classifier', config["model"])
        ])
        
        # Fit pipeline on training data
        pipeline.fit(X_train, y_train)
        
        # Print selected features
        selected_mask = pipeline.named_steps['feature_selection'].get_support()
        selected_features = X.columns[selected_mask].tolist()
        print(f"Selected features for {model_name}: {selected_features}")
        
        # Predict
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"Balanced Accuracy: {bal_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall (Sensitivity): {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        # 5-fold cross validation for evaluation (using StratifiedKFold to maintain class balance)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=['balanced_accuracy', 'recall', 'f1'])
        cv_bal_acc = cv_scores['test_balanced_accuracy'].mean()
        cv_recall = cv_scores['test_recall'].mean()
        cv_f1 = cv_scores['test_f1'].mean()
        print(f"5-Fold CV Mean Balanced Accuracy: {cv_bal_acc:.4f}")
        print(f"5-Fold CV Mean Recall: {cv_recall:.4f}")
        print(f"5-Fold CV Mean F1: {cv_f1:.4f}")
        
        results[model_name] = {
            "pipeline": pipeline,
            "metrics": {
                "balanced_accuracy": bal_acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cv_balanced_accuracy": cv_bal_acc,
                "cv_recall": cv_recall,
                "cv_f1": cv_f1
            },
            "confusion_matrix": cm
        }
        
        # Log to MLflow
        with mlflow.start_run(run_name=model_name):
            # Log params
            mlflow.log_params(config["params"])
            mlflow.log_param("scaling", "StandardScaler")
            mlflow.log_param("feature_selection", "RandomForest-SelectFromModel")
            mlflow.log_param("selected_features_count", len(selected_features))
            
            # Log metrics
            mlflow.log_metric("balanced_accuracy", bal_acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("cv_mean_balanced_accuracy", cv_bal_acc)
            mlflow.log_metric("cv_mean_recall", cv_recall)
            mlflow.log_metric("cv_mean_f1", cv_f1)
            mlflow.log_metric("tn", float(tn))
            mlflow.log_metric("fp", float(fp))
            mlflow.log_metric("fn", float(fn))
            mlflow.log_metric("tp", float(tp))
            
            # Tags
            mlflow.set_tag("model_family", config["family"])
            
            # Artifacts
            # Save confusion matrix plot as PNG
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Disease"])
            disp.plot(ax=ax, cmap=plt.cm.Blues)
            plt.title(f"{model_name} Confusion Matrix")
            cm_path = f"{config['family']}_confusion_matrix.png"
            plt.savefig(cm_path)
            plt.close()
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)
            
            # Save selected features list
            features_file = f"{config['family']}_selected_features.txt"
            with open(features_file, "w") as f:
                f.write("\n".join(selected_features))
            mlflow.log_artifact(features_file)
            os.remove(features_file)
            
            # Log model
            mlflow.sklearn.log_model(
                pipeline, name="model",
                skops_trusted_types=[
                    "sklearn.calibration._CalibratedClassifier",
                    "sklearn.calibration._SigmoidCalibration"
                ]
            )
            
    # 4. Identify Best Model & Perform Hyperparameter Tuning
    # 임상 맥락에서 False Negative를 낮추는 것이 가장 중요하므로 CV Recall을 일차 기준으로 하여 가장 우수한 모델을 선정합니다.
    best_candidate_name = max(results, key=lambda k: results[k]["metrics"]["cv_recall"])
    print(f"\nBest Candidate Model for Tuning (based on CV Recall): {best_candidate_name}")
    
    # Tuner setup based on the selected best model
    best_pipeline = results[best_candidate_name]["pipeline"]
    
    param_grid = {}
    if best_candidate_name == "Logistic Regression":
        param_grid = {
            'classifier__C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__class_weight': [None, 'balanced']
        }
    elif best_candidate_name == "SVC":
        param_grid = {
            'classifier__estimator__C': [0.1, 1.0, 10.0],
            'classifier__estimator__kernel': ['linear', 'rbf'],
            'classifier__estimator__class_weight': [None, 'balanced']
        }
    elif best_candidate_name == "Random Forest":
        param_grid = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [3, 5, None],
            'classifier__class_weight': [None, 'balanced', 'balanced_subsample']
        }
        
    print(f"Performing GridSearchCV for {best_candidate_name} with parameters: {param_grid}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=best_pipeline,
        param_grid=param_grid,
        scoring='recall',
        cv=cv,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"Best hyperparameters found: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Evaluate final model
    y_pred_final = best_model.predict(X_test)
    final_bal_acc = balanced_accuracy_score(y_test, y_pred_final)
    final_precision = precision_score(y_test, y_pred_final)
    final_recall = recall_score(y_test, y_pred_final)
    final_f1 = f1_score(y_test, y_pred_final)
    final_cm = confusion_matrix(y_test, y_pred_final)
    final_tn, final_fp, final_fn, final_tp = final_cm.ravel()
    
    print(f"\n--- Final Tuned Model ({best_candidate_name}) Results ---")
    print(f"Balanced Accuracy: {final_bal_acc:.4f}")
    print(f"Precision: {final_precision:.4f}")
    print(f"Recall (Sensitivity): {final_recall:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print(f"Confusion Matrix:\n{final_cm}")
    
    # Log Tuned Model to MLflow
    with mlflow.start_run(run_name=f"Tuned_{best_candidate_name}"):
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_param("scaling", "StandardScaler")
        mlflow.log_param("feature_selection", "RandomForest-SelectFromModel")
        
        mlflow.log_metric("balanced_accuracy", final_bal_acc)
        mlflow.log_metric("precision", final_precision)
        mlflow.log_metric("recall", final_recall)
        mlflow.log_metric("f1_score", final_f1)
        mlflow.log_metric("tn", float(final_tn))
        mlflow.log_metric("fp", float(final_fp))
        mlflow.log_metric("fn", float(final_fn))
        mlflow.log_metric("tp", float(final_tp))
        
        mlflow.set_tag("model_family", models_config[best_candidate_name]["family"])
        mlflow.set_tag("tuned", "True")
        
        # Save confusion matrix plot as PNG
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=final_cm, display_labels=["Healthy", "Disease"])
        disp.plot(ax=ax, cmap=plt.cm.Blues)
        plt.title(f"Tuned {best_candidate_name} Confusion Matrix")
        cm_path = f"tuned_{models_config[best_candidate_name]['family']}_confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)
        
        # Log model
        mlflow.sklearn.log_model(
            best_model, name="model",
            skops_trusted_types=[
                "sklearn.calibration._CalibratedClassifier",
                "sklearn.calibration._SigmoidCalibration"
            ]
        )
        
    # Save final model to disk (using pickle)
    models_dir = os.path.join(current_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "final_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Saved final model to {model_path}")
    
    # Clinical justification
    print("\n--- Clinical Justification ---")
    justification = (
        "임상적 의사결정 과정에서 False Negative(위음성)는 질병을 앓고 있는 고위험 환자를 정상으로 오진하여 "
        "치료 시기를 놓치게 만드는 가장 치명적인 위험 요인입니다. 따라서 본 프로젝트에서는 Recall(재현율)을 극대화하는 모델을 우선적으로 고려했습니다. "
        f"최종 선택된 {best_candidate_name} 모델은 하이퍼파라미터 튜닝을 거쳐 Recall {final_recall:.2%}을 달성하여 환자 누락 가능성을 최소화했습니다. "
        "일부 False Positive(위양성)가 발생하더라도 추가적인 임상 정밀 검사를 통해 정상 판별이 가능하므로, "
        "생명 보존과 조기 진단이라는 임상적 맥락에서 이 모델이 최종적으로 정당화됩니다."
    )
    print(justification)

if __name__ == "__main__":
    main()