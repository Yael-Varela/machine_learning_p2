from pathlib import Path
import json

import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from dataset import scan_files, build_dataframe, dataframe_to_arrays
from model_utils import get_train_test_split

# Using our best model with the tunned hyperparameters.
 
MODEL = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=67)
MODEL_PATH = Path('final_model.joblib')
SCALER_PATH = Path('final_scaler.joblib')
WEB_DATA_PATH = Path('datos.js')
TEST_SUBSETS_DIR = Path('test_subsets_by_class')

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


def all_metrics(y_true, y_pred):
    '''
    Returns the four evaluation metrics in a dictionary, already rounded,
    so train and test can be compared side by side.
    '''
    return {
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'precision': round(float(precision_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'recall': round(float(recall_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'f1': round(float(f1_score(y_true, y_pred, average='macro')), 4),
    }


def save_web_data(y_train, train_pred, y_test, test_pred, labels):
    '''
    Writes WEB_DATA_PATH, a small JavaScript file that resultados.html
    reads directly, so no conversion step is needed. It stores the train
    and test metrics, the bias/variance diagnosis and the confusion matrix
    of the test set. Everything is computed here with sklearn so the page
    shows exactly the same numbers printed above.
    '''
    train = all_metrics(y_train, train_pred)
    test = all_metrics(y_test, test_pred)
    bias, variance, fit, gap = diagnose(train['f1'], test['f1'])
    matrix = confusion_matrix(y_test, test_pred, labels=labels)

    data = {
        'clases': [int(c) for c in labels],
        'n_train': int(len(y_train)),
        'n_test': int(len(y_test)),
        'train': train,
        'test': test,
        'gap': round(float(gap), 4),
        'bias': bias,
        'variance': variance,
        'fit': fit,
        'matriz': [[int(v) for v in row] for row in matrix],
    }

    with open(WEB_DATA_PATH, 'w') as f:
        f.write('var DATOS = ')
        json.dump(data, f)
        f.write(';\n')

    print(f'Web data saved to {WEB_DATA_PATH}')


def save_test_subsets_by_class(X_test, y_test):
    '''
    Receives the raw (unscaled) test features and labels, splits them by
    class, and saves one .npz file per class, so an interface can demo one
    exercise at a time. This is only for the demo'''
    TEST_SUBSETS_DIR.mkdir(exist_ok=True)
    classes = np.unique(y_test)

    for exercise in classes:
        mask = y_test == exercise
        X_exercise = X_test[mask]
        y_exercise = y_test[mask]

        out_path = TEST_SUBSETS_DIR / f'class_{exercise:02d}.npz'
        np.savez(out_path, X=X_exercise, y=y_exercise)
        print(f'  class {exercise:02d}: {X_exercise.shape[0]} samples -> {out_path}')


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
    print(f'Z-Score: {outliers} values clipped in train ({outliers / X_train.size * 100:.4f}%)')
    X_train = np.clip(X_train, lower, upper)
    X_test = np.clip(X_test, lower, upper)
 
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MODEL
    model.fit(X_train_scaled, y_train)

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    train_f1 = f1_score(y_train, train_pred, average='macro')
    test_f1 = f1_score(y_test, test_pred, average='macro')

    print(f'{"":<12}{"accuracy":>10}{"f1_macro":>10}')
    print(f'{"train":<12}{train_acc:>10.4f}{train_f1:>10.4f}')
    print(f'{"test":<12}{test_acc:>10.4f}{test_f1:>10.4f}')

    bias, variance, fit, gap = diagnose(train_f1, test_f1)
    print(f'\ntrain-test f1 gap: {gap:.4f}')
    print(f'bias: {bias}')
    print(f'variance: {variance}')
    print(f'fit: {fit}')

    # full test set together (all 15 classes), not split apart -- this is
    # what tells you WHICH classes get confused with which
    print('\nConfusion matrix (rows=true, columns=predicted):')
    labels = sorted(np.unique(y))
    print(confusion_matrix(y_test, test_pred, labels=labels))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f'\nModel saved to {MODEL_PATH}')
    print(f'Scaler saved to {SCALER_PATH}')

    save_web_data(y_train, train_pred, y_test, test_pred, labels)

    print(f'\nSaving test set split by class to {TEST_SUBSETS_DIR}/')
    save_test_subsets_by_class(X_test, y_test)

if __name__ == '__main__':
    main()
