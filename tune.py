from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from dataset import scan_files, build_dataframe, dataframe_to_arrays
from model_utils import get_models, make_pipeline

'''
Hyperparameter search (GridSearchCV) for random_forest and svm_rbf, the two
models that performed best in the comparation between models. For each model it plots
how the mean cross validated F1 changes across its hyperparameter values.
'''

PARAM_GRIDS = {
    'random_forest': {
        'model__n_estimators': [100, 200, 300, 400, 500],
        'model__max_depth': [10, 20, None],
    },
    'svm_rbf': {
        'model__C': [0.1, 1, 10, 100],
        'model__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    },
}

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
PLOTS_DIR = Path('plots')


def save_plot(filename):
    '''Saves the current matplotlib figure to PLOTS_DIR and displays it.'''
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_random_forest(results, name):
    '''Plots F1 vs number of trees, with one line per max_depth value.'''
    results['trees'] = results['param_model__n_estimators'].astype(int)
    results['max_depth'] = results['param_model__max_depth'].astype(str)

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=results, x='trees', y='mean_test_score', hue='max_depth', marker='o')
    plt.xlabel('Number of trees')
    plt.ylabel('Average macro F1 (CV)')
    plt.title(f'{name}: F1 vs number of trees')
    save_plot(f'{name}_f1_vs_trees.png')


def plot_svm(results, name):
    '''Plots F1 vs C, with one line per gamma value.'''
    results['C'] = results['param_model__C'].astype(float)
    results['gamma'] = results['param_model__gamma'].astype(str)

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=results, x='C', y='mean_test_score', hue='gamma', marker='o')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('Average macro F1 (CV)')
    plt.title(f'{name}: F1 vs C')
    save_plot(f'{name}_f1_vs_C.png')


PLOT_FUNCTIONS = {
    'random_forest': plot_random_forest,
    'svm_rbf': plot_svm,
}


def tune_model(name, grid_params, X, y):
    '''Runs GridSearchCV for a single model and returns its cv_results_ as a dataframe.'''
    grid = GridSearchCV(
        make_pipeline(get_models()[name]), grid_params,
        scoring='f1_macro', cv=CV, refit=True,
    )

    start = time.perf_counter()
    grid.fit(X, y)
    elapsed = time.perf_counter() - start

    print(f'best parameters: {grid.best_params_}')
    print(f'best f1_macro: {grid.best_score_:.4f}')
    print(f'search time: {elapsed:.1f}s\n')

    return pd.DataFrame(grid.cv_results_)


def main():
    sns.set_theme(style='whitegrid')
    PLOTS_DIR.mkdir(exist_ok=True)

    movements = scan_files(Path('d02_processed_data'))
    df = build_dataframe(movements)
    X, y = dataframe_to_arrays(df)
    print(f'Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes\n')

    for name, grid_params in PARAM_GRIDS.items():
        print(f'GridSearchCV: {name}')
        results = tune_model(name, grid_params, X, y)
        PLOT_FUNCTIONS[name](results, name)

    print(f'Plots saved to {PLOTS_DIR}/')


if __name__ == '__main__':
    main()