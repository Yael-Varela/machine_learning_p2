
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from dataset import scan_files, build_dataframe, dataframe_to_arrays
from model_utils import get_train_test_split

FINAL_MODEL = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=67)

MODEL_PATH = Path('final_model.joblib')
SCALER_PATH = Path('final_scaler.joblib')
FULL_TEST_PATH = Path('full_test_predictions.npz')
TEST_SUBSETS_DIR = Path('test_subsets_by_class')

# thresholds used only to turn the train/test gap into a readable label;
# adjust if your own results call for a different cutoff
BIAS_THRESHOLD = 0.85   # below this train score -> considered high bias
GAP_THRESHOLD_LOW = 0.03
GAP_THRESHOLD_HIGH = 0.08


def diagnose(train_f1, test_f1):
    '''Turns train/test F1 scores into a bias/variance/fit diagnosis.'''
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


def save_test_subsets_by_class(X_test, y_test):
    '''
    Splits the held-out test set into one subset per class (movement_id)
    and saves each as a separate .npz file (RAW, unscaled features), so an
    interface can load one class at a time, apply the saved scaler itself,
    and run predictions on it.
    '''
    TEST_SUBSETS_DIR.mkdir(exist_ok=True)
    classes = np.unique(y_test)

    for cls in classes:
        mask = y_test == cls
        X_cls = X_test[mask]
        y_cls = y_test[mask]

        out_path = TEST_SUBSETS_DIR / f'class_{cls:02d}.npz'
        np.savez(out_path, X=X_cls, y=y_cls)
        print(f'  class {cls:02d}: {X_cls.shape[0]} samples -> {out_path}')


def main():
    movements = scan_files(Path('d02_processed_data'))
    df = build_dataframe(movements)
    X, y = dataframe_to_arrays(df)

    # same seed/test_size as the other scripts -> identical split, so this
    # test set was never seen during model comparison or tuning
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    print(f'Train: {X_train.shape[0]} samples   Test: {X_test.shape[0]} samples (held out, unseen until now)\n')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = FINAL_MODEL
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

    # ---- save everything the interface needs ----

    # trained model + scaler, so the interface can load them without retraining
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f'\nModel saved to {MODEL_PATH}')
    print(f'Scaler saved to {SCALER_PATH}')

    # predictions on the FULL test set: required to compute correct
    # precision/recall per class later (precision needs to know about
    # false positives coming from OTHER classes, which are invisible if
    # you only look at one class's subset in isolation)
    np.savez(FULL_TEST_PATH, y_true=y_test, y_pred=test_pred)
    print(f'Full test set predictions saved to {FULL_TEST_PATH}')

    # RAW (unscaled) test set split by class, for the interface to demo
    # predictions one exercise at a time
    print(f'\nSaving test set split by class to {TEST_SUBSETS_DIR}/')
    save_test_subsets_by_class(X_test, y_test)


if __name__ == '__main__':
    main()