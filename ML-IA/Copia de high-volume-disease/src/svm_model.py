# src/svm_model.py

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import joblib
import pandas as pd
import os
from sklearn.impute import SimpleImputer


def train_svm_model(df_clean):
    
    #Preparación de datos 
    X = df_clean.select_dtypes(include=['number', 'bool']).drop(columns=['Enfermedad_gran_volumen'])
    y = df_clean['Enfermedad_gran_volumen'].astype(int)  # Asegurar que la variable objetivo sea numérica


    # División del conjunto
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 30)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Modelo SVM con kernel RBF (no lineal)
    model = SVC(kernel='rbf', probability=True, class_weight='balanced')
    model.fit(X_train, y_train)

    # Guardar el modelo
    joblib.dump(model, './models/svm_target_bin.pkl')

    # Predicción
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluación
    acc = accuracy_score(y_test, y_pred)
    print(f'\nSVM - Accuracy: {acc:.2f}')
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))


    # -------------------------------
    # MÉTRICAS DE CLASIFICACIÓN
    # -------------------------------

    # Reporte de métricas
    print('\nReporte de Clasificación:')
    print(classification_report(y_test, y_pred))

    # Cálculo de métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Mostrar métricas
    print("\nMétricas del modelo DC:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    # Guardar métricas
    metricas_svm = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })
    metricas_svm.to_csv('./models/metricas_svm.csv', index=False)

    metrics = {
        'modelo': 'SVM',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Oranges')
    plt.title('Matriz de Confusión - SVM')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./models/conf_matrix_svm.png', dpi=300)
    #plt.close()
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='orange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - SVM')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./models/roc_svm.png', dpi=300)
    plt.show()

    # 🔽 Exportar curva ROC y AUC
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    roc_df.to_csv('./models/roc_svm.csv', index=False)

    with open('./models/roc_svm_auc.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')

    return model, metrics