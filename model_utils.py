'''
Shared definitions and functions for the models: 
we will first compare all models using standard parameters, and after identifying the most promising ones, 
we will fine-tune them to determine which model offers the best classification performance for the project.
'''

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

SCORING = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

def get_train_test_split(X, y, test_size=0.2, seed=42):
     return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


# Returns the available untrained models in a dictionary
def get_models():
    return {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'decision_tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=300, random_state=42),
        'svm_rbf': SVC(kernel='rbf', C=1.0, gamma='scale'),
        # regularización L2 (Ridge)
        'logistic_regression': LogisticRegression(max_iter=2000, penalty='l2', C=1.0),
        'naive_bayes': GaussianNB(),
    }

# Builds a pipeline with scaling, optional feature selection, and model.
def make_pipeline(model, n_features_to_select=None):
    steps = [('scaler', StandardScaler())]
    if n_features_to_select is not None:
        steps.append(('select', SelectKBest(score_func=f_classif, k=n_features_to_select)))
    steps.append(('model', model))
    return Pipeline(steps)

'''
Evaluates the model using stratified cross validation.
Returns mean metrics and fit time.
'''
def evaluate_with_cv(pipeline, X, y, cv=5):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    cv_results = cross_validate(pipeline, X, y, cv=skf, scoring=SCORING)
    return {
        'accuracy': cv_results['test_accuracy'].mean(),
        'precision': cv_results['test_precision_macro'].mean(),
        'recall': cv_results['test_recall_macro'].mean(),
        'f1': cv_results['test_f1_macro'].mean(),
        'fit_time': cv_results['fit_time'].mean(),
    }