import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import pipeline  # Import your custom pipeline module
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

def load_data(file_path):
    # Load the dataset from the given file path
    df = pd.read_csv(file_path, encoding='UTF-8')
    return df

def preprocess_data(df):
    if 'Game Type' in df.columns and 'Week' in df.columns:
        df = df.drop(['Unnamed: 0', 'Participant ID', 'Emotional', 'Psychological', 'Social', 'Game Type', 'Week'], axis=1)
    else:
        df = df.drop(['Unnamed: 0', 'Participant ID', 'Emotional', 'Psychological', 'Social'], axis=1)
    
    for step in preproc:
        df = df.pipe(step)
    
    return df

def train_and_evaluate_model(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    return accuracy, conf_matrix, classification_rep

def perform_cross_validation(X, y, model, cv=5, scoring='accuracy'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean(), scores.std()

def analyze_feature_importance(X, y, model, importance_method='permutation', scoring='accuracy', n_top=1):
    if importance_method == 'permutation':
        result = permutation_importance(model, X, y, n_repeats=30, random_state=0, scoring=scoring)

        importance_dict = {col: imp for col, imp in zip(X.columns, result.importances_mean)}
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

        selected_features = [col for col, imp in sorted_importance[:n_top]]

        return X[selected_features]
    else:
        raise ValueError("Unknown importance method")

if __name__ == '__main__':
    file_path = r"C:\Users\User\Desktop\SG_walk\cache_data\participant_data_all.csv"
    
    # Load the dataset
    df = load_data(file_path)

    # Define the preprocessing steps
    preproc = [
        pipeline.removeLowVar,
        pipeline.oneHotEncode,
        pipeline.upSample
    ]

    # Preprocess the data
    df = preprocess_data(df)

    # Define the models and their names
    models = {
        'RandomForest': RandomForestClassifier(),
        'Naive Bayes' : GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=20)
    }

    for model_name, model in models.items():
        X = df.drop(['flourishing', 'moderate'], axis=1)
        y = df['flourishing']

        # Train and evaluate the model
        accuracy, conf_matrix, classification_rep = train_and_evaluate_model(X, y, model)

        # Perform cross-validation
        mean_cv_score, std_cv_score = perform_cross_validation(X, y, model, cv=5, scoring='accuracy')

        # Select important features if mean accuracy is too low
        if mean_cv_score <= 0.7:
            # Analyze feature importance and select features
            selected_features = analyze_feature_importance(X, y, model, importance_method='permutation', scoring='accuracy', n_top=10)

            # Train and evaluate the model
            accuracy, conf_matrix, classification_rep = train_and_evaluate_model(selected_features, y, model)

            # Perform cross-validation
            mean_cv_score, std_cv_score = perform_cross_validation(selected_features, y, model, cv=5, scoring='accuracy')

        print("-----------Evaluation of {} model-----------".format(model))
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", classification_rep)
        print(f"Cross-Validation Accuracy (Mean): {mean_cv_score}")
        print(f"Cross-Validation Accuracy (Standard Deviation): {std_cv_score}")
        print("\n")
