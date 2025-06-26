# src/decision_tree_model.py

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import joblib
import pandas as pd
#import os
from sklearn.tree import export_graphviz
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

#import graphviz

def train_decision_tree_model(df_clean):

    #Preparación de datos
    X = df_clean.select_dtypes(include=['number', 'bool']).drop(columns=['Enfermedad_gran_volumen'])
    y = df_clean['Enfermedad_gran_volumen'].astype(int)  # Asegurar que la variable objetivo sea numérica

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Modelo
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    # Guardar el modelo
    joblib.dump(model, './models/arbol_decision_target_bin.pkl')

    # Predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Métricas
    acc = accuracy_score(y_test, y_pred)
    print(f'\n📊 Árbol de Decisión - Accuracy: {acc:.2f}')
    print('\n📑 Classification Report:')
    print(classification_report(y_test, y_pred))

    # -------------------------------
    # MÉTRICAS DE CLASIFICACIÓN
    # -------------------------------

    # Reporte de métricas
    print('\nReporte de Clasificación DC:')
    print(classification_report(y_test, y_pred))

    # Cálculo de métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Mostrar métricas
    print("\nMétricas del modelo:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Guardar métricas
    metricas_tree = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })
    metricas_tree.to_csv('./models/metricas_tree.csv', index=False)

    metrics = {
        'modelo': 'Arbol de decisión',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }



    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Greens')
    plt.title('Matriz de Confusión - Árbol de Decisión')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./models/conf_matrix_tree.png', dpi=300)
    #plt.close()
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='green')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - Árbol de Decisión')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./models/roc_tree.png', dpi=300)
    plt.show()

    # Visualización del árbol
    plt.figure(figsize=(16, 8))
    #plot_tree(model, feature_names=features, class_names=['No', 'Sí'], filled=True)
    plot_tree(model, feature_names=X.columns, class_names=model.classes_.astype(str), filled=True)

    plt.title('Árbol de Decisión (max_depth=4)')
    plt.tight_layout()
    plt.savefig('./models/tree.png', dpi=300)
    plt.show()

    joblib.dump(X.columns.tolist(), './models/arbol_decision_columns.pkl')

   
     # 🔽 Guardar curva ROC y AUC
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    roc_df.to_csv('./models/roc_arbol.csv', index=False)

    with open('./models/roc_arbol_auc.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')
    

    # Evaluar cada modelo
    results_tree = {}
    name = 'Árbol de Decisión'

    #Resultados modelo
    print('Resultados Modelo Árbol de Decisión')
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results_tree[name] = {
        "Accuracy": acc,
        "F1-score (class 1)": report["1"]["f1-score"],
        "Precision (class 1)": report["1"]["precision"],
        "Recall (class 1)": report["1"]["recall"]
    }
    
    return model, metrics