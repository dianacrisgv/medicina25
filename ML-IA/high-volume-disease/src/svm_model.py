# src/svm_model.py

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import joblib
import pandas as pd
import os

def train_svm_model(df, target='target_bin'):
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # Definir columnas a eliminar dinámicamente
    drop_candidates = [target, 'Enfermedad_gran_volumen', 'Documento','Genero', 'Fecha Nacimiento','Edad','DXprincipal', 'Des_Dx','Anoscopia', 'Sifilis', 'proctitis', 'Genotipos' ]

    cols_to_drop = [col for col in drop_candidates if col in df.columns]

    features = df.drop(columns=cols_to_drop).columns.tolist()
    X = df[features]
    X = pd.get_dummies(X, drop_first=True)  # Convierte texto a variables dummy
    y = df[target]

    # División del conjunto
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Oranges')
    plt.title('Matriz de Confusión - SVM')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./models/conf_matrix_svm.png')
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
    plt.show()

    # 🔽 Exportar curva ROC y AUC
    roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr})
    roc_df.to_csv('./models/roc_svm.csv', index=False)

    with open('./models/roc_svm_auc.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')