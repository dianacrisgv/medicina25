import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import joblib

def train_svm_model(df, target='target_bin'):
    # Definir columnas a eliminar
    drop_candidates = [target, 'num']
    cols_to_drop = [col for col in drop_candidates if col in df.columns]

    features = df.drop(columns=cols_to_drop).columns.tolist()
    X = df[features]
    y = df[target]

    # División del conjunto
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo SVM con kernel RBF (no lineal)
    model = SVC(kernel='rbf', probability=True)
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
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusión - SVM')
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', color='purple')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC - SVM')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()