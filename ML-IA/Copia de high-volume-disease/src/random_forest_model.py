import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import joblib
import pandas as pd
import os
from sklearn.impute import SimpleImputer


def train_random_forest_model(df_clean):
    
    #Preparación de datos
    X = df_clean.select_dtypes(include=['number', 'bool']).drop(columns=['Enfermedad_gran_volumen'])
    y = df_clean['Enfermedad_gran_volumen'].astype(int)  # Asegurar que la variable objetivo sea numérica

    # 🔹 División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # 🔹 Entrenamiento del modelo
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    # 🔹 Guardar el modelo entrenado
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model, './models/random_forest_target_bin.pkl')

    # 🔹 Predicciones
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 🔹 Evaluación
    acc = accuracy_score(y_test, y_pred)
    print(f'\n🌲 Random Forest - Accuracy: {acc:.2f}')
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
    metricas_rf = pd.DataFrame({
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1_score': [f1]
    })
    metricas_rf.to_csv('./models/metricas_rf.csv', index=False)

    metrics = {
        'modelo': 'Random forest',
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }


    # 🔹 Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Purples')
    plt.title('Matriz de Confusión - Random Forest')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./models/conf_matrix_rf.png', dpi=300)
    #plt.close()
    plt.show()

    # 🔹 Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='purple')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - Random Forest')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./models/roc_rf.png', dpi=300)
    plt.show()

    # 🔹 Guardar curva ROC y AUC
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv('./models/roc_rf.csv', index=False)
    with open('./models/roc_rf_auc.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')

    return model, metrics