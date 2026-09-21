# REHAB Model Selection
 
`compare_models.py` compara el rendimiento de los 6 modelos candidatos (accuracy, precision, recall, F1, tiempo de entrenamiento), usando cross-validation sobre el 80% de entrenamiento.
 
Con base en esos resultados, se eligieron Random Forest y SVM para `tune.py`, que busca sus mejores hiperparámetros haciendo uso de `GridSearchCV` y genera las gráficas de F1 vs. hiperparámetros (guardados en folds si se desean ver).
 
`evaluate_final_model.py` entrena el modelo final (Random Forest, con los hiperparámetros ya seleccionados) sobre el 100% del 80% de entrenamiento, lo evalúa contra el 20% de test que nunca se había visto antes. 
Finalmente, los resutlados se pueden observar en el archivo de pagina web .html.
