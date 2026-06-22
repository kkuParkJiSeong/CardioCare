import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# 결측값 처리
def impute_values(df, target_cols=['ca', 'thal']):
    df_impute = df.copy()
    imputer = SimpleImputer(strategy='most_frequent')
    df_impute[target_cols] = imputer.fit_transform(df[target_cols])

    return df_impute
# 이상치 처리
def remove_outlier(df):
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df
# 빈 컬럼 제거
def remove_empty_columns(df, threshold=0.5):
    null_percent = df.isnull().mean()
    cols_to_drop = null_percent[null_percent >= threshold].index.tolist()
    return df.drop(columns=cols_to_drop)
    
# 중복 제거
def remove_duplicate_row(df):
    return df.drop_duplicates().reset_index(drop=True)

def validate_ranges(df):
    if 'chol' in df.columns:
        # Check if any value is outside [0, 600]
        if not df['chol'].between(0, 600).all():
            raise ValueError("Cholesterol (chol) values must be between 0 and 600.")
    if 'age' in df.columns:
        if not df['age'].between(0, 120).all():
            raise ValueError("Age values must be between 0 and 120.")
    return True

def preprocessing_data(df):
    df = remove_empty_columns(df)
    df = remove_outlier(df)
    df = impute_values(df)
    df = remove_outlier(df)
    validate_ranges(df)
    return df