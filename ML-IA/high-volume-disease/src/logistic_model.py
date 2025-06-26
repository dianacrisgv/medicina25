import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Importar las siguientes métricas
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

import joblib
import pandas as pd
import os

from sklearn.impute import SimpleImputer


def train_logistic_model(df_clean):
    
    #Preparación de datos
    X = df_clean.select_dtypes(include=['number', 'bool']).drop(columns=['Enfermedad_gran_volumen'])
    y = df_clean['Enfermedad_gran_volumen'].astype(int)  # Asegurar que la variable objetivo sea numérica
    
    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Entrenamiento del modelo de regresión logística
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)

    # Guardar el modelo
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model, './models/regresion_logistica_target_bin.pkl')

    # Predicción
    y_pred = model.predict(X_test)

    # Evaluación básica
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nEvaluación del modelo de regresión logística DC:')
    print(f'Accuracy: {accuracy:.2f}')


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
    metricas_log = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })
    metricas_log.to_csv('./models/metricas_logistic.csv', index=False)

    metrics = {
        'modelo': 'Regresión logística',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }


    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusión - Regresión Logística')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./models/conf_matrix_logistic.png', dpi=300)
    #plt.close()
    plt.show()

    # Curva ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - Regresión Logística')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./models/roc_log.png', dpi=300)
    plt.show()


    # 🔽 Exportar curva ROC y AUC
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    roc_df.to_csv('./models/roc_log.csv', index=False)

    with open('./models/roc_logistic_auc.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')

    

    return model, metrics