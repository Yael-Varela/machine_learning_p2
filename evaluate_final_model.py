from pathlib import Path
import json

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, confusion_matrix)
from dataset import scan_files, build_dataframe, dataframe_to_arrays
from model_utils import get_train_test_split

# Using our best model with the tunned hyperparameters.
MODEL = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=67)
WEB_DATA_PATH = Path('datos.js')
CURVE_STEPS = 10   # how many points the learning curves have

# thresholds used only to turn the train/test gap into a readable label
BIAS_THRESHOLD = 0.85   # below this will be consider as 'high bias'
GAP_THRESHOLD_LOW = 0.03
GAP_THRESHOLD_HIGH = 0.08


def diagnose(train_f1, test_f1):
    '''
    Receives the train and test F1 scores, computes the gap between them,
    and classifies bias, variance and overall fit from that gap. Returns
    (bias, variance, fit, gap).
    '''
    gap = train_f1 - test_f1

    if train_f1 < BIAS_THRESHOLD:
        bias = 'alto'
    elif train_f1 < BIAS_THRESHOLD + 0.1:
        bias = 'medio'
    else:
        bias = 'bajo'

    if gap < GAP_THRESHOLD_LOW:
        variance = 'baja'
    elif gap < GAP_THRESHOLD_HIGH:
        variance = 'media'
    else:
        variance = 'alta'

    if bias == 'alto':
        fit = 'underfit'
    elif variance == 'alta':
        fit = 'overfit'
    else:
        fit = 'good fit'

    return bias, variance, fit, gap


def squared_error(model, X, y):
    '''
    Mean squared error between the probabilities the model gives to each
    class and the correct answer (1 for the true class, 0 for the rest).
    This is how the error is measured for a classifier, since the labels
    are not numbers you can subtract.
    '''
    proba = model.predict_proba(X)
    target = np.zeros_like(proba)
    positions = {label: i for i, label in enumerate(model.classes_)}
    for row, label in enumerate(y):
        target[row, positions[label]] = 1
    return float(np.mean(np.sum((proba - target) ** 2, axis=1)))


def learning_curves(X_train, y_train, X_test, y_test):
    '''
    Trains the same model again and again with a growing slice of the
    training set, and measures F1 and error on train and test every time.
    This is what shows how far apart the two curves stay and whether more
    data would still help.
    '''
    sizes, f1_train, f1_test, error_train, error_test = [], [], [], [], []

    for step in range(1, CURVE_STEPS + 1):
        fraction = step / CURVE_STEPS

        # a stratified slice, so every exercise keeps its proportion
        if fraction < 1:
            index, _ = train_test_split(np.arange(len(y_train)), train_size=fraction, random_state=67, stratify=y_train)
        else:
            index = np.arange(len(y_train))

        X_part, y_part = X_train[index], y_train[index]

        model = clone(MODEL)
        model.fit(X_part, y_part)

        sizes.append(int(len(y_part)))
        f1_train.append(round(float(f1_score(y_part, model.predict(X_part), average='macro')), 4))
        f1_test.append(round(float(f1_score(y_test, model.predict(X_test), average='macro')), 4))
        error_train.append(round(squared_error(model, X_part, y_part), 4))
        error_test.append(round(squared_error(model, X_test, y_test), 4))

        print(f'{len(y_part):>8}{f1_train[-1]:>12.4f}{f1_test[-1]:>10.4f}'
              f'{error_train[-1]:>12.4f}{error_test[-1]:>10.4f}')

    return {
        'sizes': sizes,
        'f1_train': f1_train, 'f1_test': f1_test,
        'error_train': error_train, 'error_test': error_test,
    }


def all_metrics(y_true, y_pred):
    '''
    Returns the four evaluation metrics in a dictionary, already rounded,
    plus the raw count of correct and incorrect predictions, so train and
    test can be compared side by side.
    '''
    correct = int((np.asarray(y_true) == np.asarray(y_pred)).sum())
    total = len(y_true)
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, y_pred, average='macro')), 4),
        'correct': correct,
        'incorrect': total - correct,
        'total': total,
    }


def save_web_data(train_metrics, test_metrics, y_test, test_pred, labels, curves):
    '''
    Writes WEB_DATA_PATH, a small JavaScript file that the html
    reads directly, so no conversion step is needed. It stores the train
    and test metrics (including correct/incorrect counts), the
    bias/variance diagnosis and the confusion matrix of the test set.
    Everything is computed here with sklearn so the page shows exactly the
    same numbers printed above.
    '''
    bias, variance, fit, gap = diagnose(train_metrics['f1'], test_metrics['f1'])
    matrix = confusion_matrix(y_test, test_pred, labels=labels)

    data = {
        'clases': [int(c) for c in labels],
        'n_train': train_metrics['total'],
        'n_test': test_metrics['total'],
        'train': train_metrics,
        'test': test_metrics,
        'gap': round(float(gap), 4),
        'bias': bias,
        'variance': variance,
        'fit': fit,
        'matriz': [[int(v) for v in row] for row in matrix],
        'curvas': curves,
    }

    with open(WEB_DATA_PATH, 'w') as f:
        f.write('var DATOS = ')
        json.dump(data, f)
        f.write(';\n')

    print(f'Web data saved to {WEB_DATA_PATH}')


def main():
    movements = scan_files(Path('d02_processed_data'))
    df = build_dataframe(movements)
    X, y = dataframe_to_arrays(df)

    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f'Train: {X_train.shape[0]} samples   Test: {X_test.shape[0]} samples (held out, unseen until now)\n')

   # Z-Score 
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    lower, upper = mean - 3 * std, mean + 3 * std
    outliers = ((X_train < lower) | (X_train > upper)).sum()
    print(f'Z-Score: {outliers} of {X_train.size} values clipped in train ({outliers / X_train.size * 100:.4f}%)')
    X_train = np.clip(X_train, lower, upper)
    X_test = np.clip(X_test, lower, upper)
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MODEL
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print(f'{"":<12}{"accuracy":>10}{"f1_macro":>10}{"correct":>10}{"incorrect":>12}')
    train_metrics = all_metrics(y_train, train_pred)
    test_metrics = all_metrics(y_test, test_pred)
    print(f'{"train":<12}{train_metrics["accuracy"]:>10.4f}{train_f1:>10.4f}'
          f'{train_metrics["correct"]:>10}{train_metrics["incorrect"]:>12}')
    print(f'{"test":<12}{test_metrics["accuracy"]:>10.4f}{test_f1:>10.4f}'
          f'{test_metrics["correct"]:>10}{test_metrics["incorrect"]:>12}')

    bias, variance, fit, gap = diagnose(train_f1, test_f1)
    print(f'\ntrain-test f1 gap: {gap:.4f}')
    print(f'bias: {bias}')
    print(f'variance: {variance}')
    print(f'fit: {fit}')

    # full test set together (all 15 classes), not split apart, for the confusion matrix
    print('\nConfusion matrix (rows=true, columns=predicted):')
    labels = sorted(np.unique(y))
    print(confusion_matrix(y_test, test_pred, labels=labels))

    print('\nLearning curves (retraining with more data every step):')
    print(f'{"samples":>8}{"f1 train":>12}{"f1 test":>10}{"err train":>12}{"err test":>10}')
    curves = learning_curves(X_train_scaled, y_train, X_test_scaled, y_test)

    print()
    save_web_data(train_metrics, test_metrics, y_test, test_pred, labels, curves)

if __name__ == '__main__':
    main()