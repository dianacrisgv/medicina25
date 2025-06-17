import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import joblib
import pandas as pd
import os

def train_random_forest_model(df, target='target_bin'):
    # 🔹 Columnas a eliminar (ajustar según tu dataset)
    drop_candidates = [target, 'Enfermedad_gran_volumen', 'Documento', 'Genero', 'Fecha Nacimiento',
                       'Edad', 'DXprincipal', 'Des_Dx', 'Anoscopia', 'Sifilis', 'proctitis', 'Genotipos']
    cols_to_drop = [col for col in drop_candidates if col in df.columns]

    # 🔹 Variables predictoras
    features = df.drop(columns=cols_to_drop).columns.tolist()
    X = df[features]
    X = pd.get_dummies(X, drop_first=True)  # Codificar variables categóricas
    y = df[target]

    # 🔹 División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # 🔹 Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Purples')
    plt.title('Matriz de Confusión - Random Forest')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./models/conf_matrix_rf.png')
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
    plt.show()

    # 🔹 Guardar curva ROC y AUC
    pd.DataFrame({'fpr': fpr, 'tpr': tpr}).to_csv('./models/roc_rf.csv', index=False)
    with open('./models/roc_rf_auc.txt', 'w') as f:
        f.write(f'AUC: {roc_auc:.4f}')