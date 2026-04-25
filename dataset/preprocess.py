# Não é necessário, como o dataset é gerado artificialmentel, não existem outliers, nem dados repetidos 
from . import view as view
import pandas as pd
from sklearn.model_selection import train_test_split

def analize_quantitative_attributes(df):
    view.plot_histogram(df, "screen_time_hours", 10) # bins 10 parece bom
    view.plot_histogram(df, "daily_study_hours", 10) # bins 10 parece bom
    view.plot_histogram(df, "daily_sleep_hours", 6) # bins 6 parece bom
    view.plot_histogram(df, "physical_activity_hours", 4) # bins 4 parece bom
    view.plot_histogram(df, "attendance_percentage", 10) # bins 10 parece bom
    view.plot_histogram(df, "cgpa", 6) # bins 6 parece bom

def preprocess_naive_bayes(df):
        
    df.drop(columns=["student_id"], inplace=True)  
    
    # analize_quantitative_attributes(df)
    df['screen_time_hours'] = pd.qcut(df['screen_time_hours'], q=10, labels=False, duplicates='drop')
    df['daily_study_hours'] = pd.qcut(df['daily_study_hours'], q=10, labels=False, duplicates='drop')
    df['daily_sleep_hours'] = pd.qcut(df['daily_sleep_hours'], q=6, labels=False, duplicates='drop')
    df['physical_activity_hours'] = pd.qcut(df['physical_activity_hours'], q=4, labels=False, duplicates='drop')
    df['attendance_percentage'] = pd.qcut(df['attendance_percentage'], q=10, labels=False, duplicates='drop')
    df['cgpa'] = pd.qcut(df['cgpa'], q=6, labels=False, duplicates='drop')

    
    # mapping nos q tem ordem
    df['stress_level'] = df['stress_level'].map({
        'Low': 0,
        'Medium': 1,
        'High': 2
    }).astype(int)

    df['sleep_quality'] = df['sleep_quality'].map({
        'Poor': 0,
        'Average': 1,
        'Good': 2
    }).astype(int)

    df['year'] = df['year'].map({
        '1st': 0,
        '2nd': 1,
        '3rd': 2,
        '4th': 3,  
    }).astype(int)

    # one-hot (sem ordem)
    df = pd.get_dummies(df, columns=[
        'gender',
        'course',
        'internet_quality'
    ])
    
    return df

def dataset_split(df, target_column="burnout_level", test_size=0.2, random_state=42):    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    return X_train, X_test, y_train, y_test