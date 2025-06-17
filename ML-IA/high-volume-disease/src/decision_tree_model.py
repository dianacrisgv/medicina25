# src/decision_tree_model.py

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import joblib
import pandas as pd
#import os
from sklearn.tree import export_graphviz
#import graphviz

def train_decision_tree_model(df, target='target_bin'):

    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Definir columnas a eliminar dinámicamente
    drop_candidates = [target, 'Enfermedad_gran_volumen', 'Documento','Genero', 'Fecha Nacimiento','Edad','DXprincipal', 'Des_Dx','Anoscopia', 'Sifilis', 'proctitis', 'Genotipos' ]

    cols_to_drop = [col for col in drop_candidates if col in df.columns]

    features = df.drop(columns=cols_to_drop).columns.tolist()
    X = df[features]
    X = pd.get_dummies(X, drop_first=True)  # Convierte texto a variables dummy
    y = df[target]

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Greens')
    plt.title('Matriz de Confusión - Árbol de Decisión')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./models/conf_matrix_tree.png')
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
    plt.show()

    # Visualización del árbol
    plt.figure(figsize=(16, 8))
    #plot_tree(model, feature_names=features, class_names=['No', 'Sí'], filled=True)
    plot_tree(model, feature_names=X.columns, class_names=model.classes_.astype(str), filled=True)

    plt.title('Árbol de Decisión (max_depth=4)')
    plt.tight_layout()
    plt.show()

    joblib.dump(X.columns.tolist(), './models/arbol_decision_columns.pkl')

   
     # 🔽 Guardar curva ROC y AUC
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    roc_df.to_csv('./models/roc_arbol.csv', index=False)

    with open('./models/roc_arbol_auc.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')