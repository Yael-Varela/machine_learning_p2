from pathlib import Path
import numpy as np
import pandas as pd
from dataset import scan_files, build_dataframe, dataframe_to_arrays
from model_utils import get_models, make_pipeline, evaluate_with_cv, get_train_test_split

# Cross validation comparison of 6 classical models (accuracy, precision, recall, f1, fit_time).
def main():
    movements = scan_files(Path('d02_processed_data'))
    df = build_dataframe(movements)
    print(df.head())
    X, y = dataframe_to_arrays(df)
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)
    
    print(f'Dataset: {X.shape[0]} samples total ({X_train.shape[0]} train / {X_test.shape[0]} test held out), {X.shape[1]} features, {len(np.unique(y))} classes\n')
    
    rows = []
    for name, model in get_models().items():
        pipeline = make_pipeline(model)
        metrics = evaluate_with_cv(pipeline, X_train, y_train)
        metrics['model'] = name
        rows.append(metrics)
        print(f"{name:<22} f1={metrics['f1']:.4f}  acc={metrics['accuracy']:.4f}  "
              f"precision={metrics['precision']:.4f}  recall={metrics['recall']:.4f}  "
              f"fit_time={metrics['fit_time']:.4f}s")

    results_df = pd.DataFrame(rows).sort_values('f1', ascending=False)
    print('\nRanked by f1_macro:')
    print(results_df[['model', 'f1', 'accuracy', 'precision', 'recall', 'fit_time']].to_string(index=False))


if __name__ == '__main__':
    main()