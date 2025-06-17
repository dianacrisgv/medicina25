import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
import pandas as pd
import os

def train_xgboost_model(df, target='target_bin'):
    # Definir columnas a eliminar dinámicamente
    drop_candidates = [target, 'Enfermedad_gran_volumen', 'Documento','Genero', 'Fecha Nacimiento','Edad','DXprincipal', 'Des_Dx','Anoscopia', 'Sifilis', 'proctitis', 'Genotipos' ]
    cols_to_drop = [col for col in drop_candidates if col in df.columns]

    # Variables predictoras
    features = df.drop(columns=cols_to_drop).columns.tolist()
    X = df[features]
    X = pd.get_dummies(X, drop_first=True)
    y = df[target]

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo XGBoost
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Guardar modelo
    os.makedirs('./models', exist_ok=True)
    joblib.dump(model, './models/xgboost_target_bin.pkl')

    # Predicción
    y_pred = model.predict(X_test)

    # Evaluación
    accuracy = accuracy_score(y_test, y_pred)
    print(f'\nEvaluación del modelo XGBoost:')
    print(f'Accuracy: {accuracy:.2f}')

    print('\nReporte de Clasificación:')
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusión - XGBoost')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./models/conf_matrix_xgboost.png')
    plt.show()

    # Curva ROC
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - XGBoost')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Visualización predicción vs realidad
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.xlabel('Valor Real')
    plt.ylabel('Predicción')
    plt.title('XGBoost: Clasificación binaria')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Exportar curva ROC y AUC
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    roc_df.to_csv('./models/roc_xgboost.csv', index=False)

    with open('./models/roc_xgboost_auc.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')