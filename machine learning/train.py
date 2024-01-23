'''
Important questions:
1. Should the wellbeing scores be used instead since the predicted class for one participant will always be the same
2. Should the temporal dependencies be considered
'''

import pipeline

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier 

import pandas as pd
from tqdm import tqdm
import time
import numpy as np
 
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

def load_data(file_path):
    # Load the dataset from the given file path
    df = pd.read_csv(file_path, encoding='UTF-8', index_col=0)
    return df

def preprocess_data(df):
    # Define the preprocessing steps
    preproc = [
        # pipeline.upSample,
        pipeline.scaleContVar,
        pipeline.removeLowVar
    ]

    # Remove identity columns & columns used for deriving target variable
    if 'Game Type' in df.columns and 'Week' in df.columns:
        df = df.drop(['Participant ID', 'Emotional', 'Psychological', 'Social', 'Game Type', 'Week'], axis=1)
    else:
        df = df.drop(['Participant ID', 'Emotional', 'Psychological', 'Social'], axis=1)
    
    # Preprocess the data
    for step in preproc:
        df = df.pipe(step)
    
    return df

def train_and_evaluate(X_train, X_test, y_train, y_test, model):

    print(f"Training model with features: {X.columns}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    class_report = {'accuracy' : acc, 'precision' : prec, 'recall' : recall, 'f1_score' : f1}

    return class_report, y_pred

def perform_cross_validation(X, y, model, cv=5, scoring='accuracy'):
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return scores.mean(), scores.std()
    
def hyper_parameter_tuning(clf):
    # define random parameters grid
    n_estimators = [10, 15, 20, 25] # number of trees in the random forest
    max_features = ['sqrt', 'log2'] # number of features in consideration at every split
    max_depth = [int(x) for x in np.linspace(20, 40, num = 5)] # maximum number of levels allowed in each decision tree
    min_samples_split = [6, 10, 12] # minimum sample number to split a node
    min_samples_leaf = [3, 5, 7] # minimum sample number that can be stored in a leaf node

    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                  }
    
    from sklearn.model_selection import RandomizedSearchCV
    model_tuning = RandomizedSearchCV(estimator = clf, param_distributions = random_grid,
                   n_iter = 100, cv = 20, verbose=1, random_state=21, n_jobs = -1)
    model_tuning.fit(X_train, y_train)

    print ('Random grid: ', random_grid, '\n')
    # print the best parameters
    print ('Best Parameters: ', model_tuning.best_params_, ' \n')

    best_params = model_tuning.best_params_
    n_estimators = best_params['n_estimators']
    min_samples_split = best_params['min_samples_split']
    min_samples_leaf = best_params['min_samples_leaf']
    max_features = best_params['max_features']
    max_depth = best_params['max_depth']
    
    if type(clf).__name__ == 'RandomForestClassifier':
        model_tuned = RandomForestClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf, max_features=max_features,
                        max_depth=max_depth)
    elif type(clf).__name__ == 'GradientBoostingClassifier':
        model_tuned = GradientBoostingClassifier(n_estimators=n_estimators, min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf, max_features=max_features,
                        max_depth=max_depth)
        
    mlflow.log_params(params=best_params)

    return model_tuned
    
if __name__ == '__main__':
    file_path = r"C:\Users\User\Desktop\SG_walk\cache_data\data.csv"
    
    # Load the dataset
    df = load_data(file_path)

    # Preprocess the data
    df = preprocess_data(df)

    # Define the models and their names
    models = {
        'Random Forest': RandomForestClassifier(random_state=12, bootstrap=False, max_depth=20, n_estimators=10),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=10, learning_rate=5.0, max_depth=20, random_state=0)
    }
    
    # Intiialize dictionary for models' metrics
    eval = {keys: {} for keys, _ in models.items()}

    for model_name, model in tqdm(models.items()):
        start_time = time.time()
        print("\n")
        print("Training model {}".format(model_name))
        X = df.drop(['flourishing', 'moderate'], axis=1)
        y = df['flourishing']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Cache the test data for testing purposes
        
        with mlflow.start_run():
           
            # Training the model with optimal hyperparameters
            model = hyper_parameter_tuning(model)
            metrics, predictions = train_and_evaluate(X_train, X_test, y_train, y_test, model)
            mean_cv_score, std_cv_score = perform_cross_validation(X, y, model, cv=10, scoring='accuracy')

            end_time = time.time()
            time_taken = end_time - start_time
            for score_type, score in metrics.items():
                eval[model_name][score_type] = score
                eval[model_name][score_type] = score
                eval[model_name][score_type] = score
                eval[model_name][score_type] = score
            eval[model_name]['mean_cv_score'] = mean_cv_score
            eval[model_name]['std_cv_score'] = std_cv_score

            # Logging model information
            mlflow.log_metrics(eval[model_name])
            signature = infer_signature(X_train, predictions)

            # Log the sklearn model and register as version 1
            log_result = mlflow.sklearn.log_model(
                sk_model=model,
                conda_env = r"C:\Users\User\Desktop\SG_walk\conda.yaml",
                artifact_path=model_name,
                signature=signature,
                registered_model_name=f"sg_walk_{model_name}_model",
            )

            mlflow.register_model(f"runs:/{log_result.run_id}/{model_name}", f"sg_walk_{model_name}_model")

    print(eval)

    
       
   
    
