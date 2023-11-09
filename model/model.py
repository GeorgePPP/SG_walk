import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
import pipeline  # Import your custom pipeline module
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
from tqdm import tqdm
from matplotlib import pyplot as plt
import time
import os 

def load_data(file_path):
    # Load the dataset from the given file path
    df = pd.read_csv(file_path, encoding='UTF-8')
    return df

def preprocess_data(df):

    # Define the preprocessing steps
    preproc = [
        pipeline.upSample,
        pipeline.oneHotEncode,
        pipeline.scaleContVar,
        pipeline.removeLowVar
    ]

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
    
    # index = 1
    # for tree in model.estimators_:
    #     fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    #     plot_tree(tree,
    #                 feature_names = list(X.columns), 
    #                 filled = True)
    #     fig.savefig('rf_individualtree{}.png'.format(index))
    #     index += 2
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
        print("Selected features {}".format(selected_features))

        return X[selected_features]
    else:
        raise ValueError("Unknown importance method")
    
if __name__ == '__main__':
    file_path = r"C:\Users\User\Desktop\SG_walk\cache_data\participant_specific_data.csv"
    
    # Load the dataset
    df = load_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

    # Define the models and their names
    models = {
        'Random Forest': RandomForestClassifier(random_state=12, bootstrap=False, max_depth=2),
        'Naive Bayes' : GaussianNB(),
        'Logistic Regression': LogisticRegression(max_iter=20, solver='liblinear'),
        # 'Support Vector Machine': svm.SVC(C=100, kernel='linear')
    }
    
    for model_name, model in tqdm(models.items()):
        start_time = time.time()
        print("\n")
        print("Training model {}".format(model_name))
        X = df.drop(['flourishing', 'moderate'], axis=1)
        y = df['flourishing']

        # Train and evaluate the model
        accuracy, conf_matrix, classification_rep = train_and_evaluate_model(X, y, model)

        # Perform cross-validation
        mean_cv_score, std_cv_score = perform_cross_validation(X, y, model, cv=10, scoring='accuracy')

        # Select important features if mean accuracy is too low
        if mean_cv_score <= 0.7:
            # Analyze feature importance and select features
            selected_features = analyze_feature_importance(X, y, model, importance_method='permutation', scoring='accuracy', n_top=5)

            # Train and evaluate the model
            accuracy, conf_matrix, classification_rep = train_and_evaluate_model(selected_features, y, model)
        
            # Perform cross-validation
            mean_cv_score, std_cv_score = perform_cross_validation(selected_features, y, model, cv=5, scoring='accuracy')

        end_time = time.time()
        time_taken = end_time - start_time
        print("-----------Evaluation of {} model-----------".format(model))
        print(f"Time taken for training: {time_taken}")
        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", classification_rep)
        print(f"Cross-Validation Accuracy (Mean): {mean_cv_score}")
        print(f"Cross-Validation Accuracy (Standard Deviation): {std_cv_score}")
        print("\n")