import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from joblib import dump
import click
import json

@click.command()
@click.option('--prediction_task', type=str, required=True, help='name of the prediction task')
@click.option('--data_filename', type=str, required=True, help='location of processed data')
@click.option('--model_directory', type=str, required=True, help='directory of model outputs')
def build_classifier(prediction_task, data_filename, model_directory):

    # Load dataset
    data = pd.read_csv(data_filename)
    X = data.drop('Patient_survived', axis=1)
    y = data['Patient_survived']

    # Initialize Logistic Regression classifier (with balanced class weights)
    clf = LogisticRegression(class_weight='balanced')

    # Evaluate accuracy using 5-fold cross-validation
    accuracy = cross_val_score(clf, X, y, cv=5)

    # Evaluate AUC using 5-fold cross-validation
    auc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')

    # Print results
    print('Accuracy:', accuracy.mean())
    print('AUC:', auc.mean())

    # Save results to a JSON file
    results = {'model' : 'logistic_regression', 'accuracy': accuracy.mean(), 'auc': auc.mean()}
    with open(''.join([model_directory, prediction_task, '_results.json']), 'w') as f:
        json.dump(results, f)

    # Train classifier on the full dataset
    clf.fit(X, y)
    
    # Save the trained model to a file
    dump(clf, ''.join([model_directory, prediction_task, '_model.joblib']))

    # Print probability estimates for each class
    y_prob = clf.predict_proba(X)
    class_0_prob = y_prob[y == 0][:, 0].mean()
    class_1_prob = y_prob[y == 1][:, 1].mean()
    
    print('Average probability estimate for class 0:', class_0_prob)
    print('Average probability estimate for class 1:', class_1_prob)
    
if __name__ == '__main__':
    build_classifier()