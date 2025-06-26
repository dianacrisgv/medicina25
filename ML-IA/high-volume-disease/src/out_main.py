import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import clean_data

#from regression_model import train_regression_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from logistic_model import train_logistic_model
from decision_tree_model import train_decision_tree_model
from svm_model import train_svm_model
from random_forest_model import train_random_forest_model
from confusion_matrices import show_and_save_confusion_matrices
#from xgboost_model import train_xgboost_model

# URL del dataset
url = 'https://raw.githubusercontent.com/John624-web/curso-IA-John/refs/heads/main/IA%20Coloproctologia.csv'

# Cargar datos
df = pd.read_csv(url)

print(df.head())
df.head()
df.describe()

# Guardar los datos como CSV
df.to_csv('./data/IAColoprocto_raw.csv', index=False)

# Cargar datos
df = pd.read_csv('./data/IAColoprocto_raw.csv')

# Limpiar datos
df_clean = clean_data(df)

# Cargar versión limpia
#df_clean = pd.read_csv("./data/Enfermedad_gran_volumen_clean.csv")

print(df_clean.info())

df_clean.describe()

#Estadísticas descriptivas y tipos de variables raw
print("Primeras filas del dataset:") 
print(df_clean.head())

# Análisis descriptivo de variables numéricas
print("\nResumen estadístico variables numéricas:") 
print(df_clean.describe(include=['int64', 'float64']))

# Análisis descriptivo de variables categóricas
df_clean.describe(include=['object', 'bool'])

#print("\nTipos de datos:") 
#print(df.dtypes)

# Clasificación binaria: 0 = sin enfermedad, 1 = con enfermedad
df_clean['target_bin'] = df_clean['Enfermedad_gran_volumen'].apply(lambda x: 0 if x == 'No' else 1)
df_clean.to_csv('./data/gran_volumen_clean.csv', index=False)

print(df_clean.head())

print(df_clean.describe())


""" # Visualizar la distribución de las variables numéricas
df_clean.hist(bins=20, figsize=(14, 10), color='skyblue')
plt.suptitle('Distribuciones de variables numéricas')
plt.tight_layout()
plt.show() """

# Distribución de la variable objetivo
sns.countplot(data=df_clean, x='Enfermedad_gran_volumen', palette='pastel')
plt.title('Distribución de la Enfermedad de Gran Volumen')
plt.xlabel('Enfermedad (0 = No, 1 = Sí)')
plt.ylabel('Cantidad')
plt.show()


# Boxplot de edad por clase de enfermedad
sns.boxplot(data=df_clean, x='Enfermedad_gran_volumen', y='Edad', color='skyblue')
plt.title('Edad según presencia de Enfermedad de Gran Volumen')
plt.xlabel('Enfermedad (0 = No, 1 = Sí)')
plt.ylabel('Edad')
plt.show()

# Boxplot de CD4 y CD8 por clase de enfermedad
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df_clean, x='Enfermedad_gran_volumen', y='CD4', ax=axes[0], color='skyblue')
axes[0].set_title('CD4 según enfermedad')
axes[0].set_xlabel('Enfermedad')
axes[0].set_ylabel('CD4')

sns.boxplot(data=df_clean, x='Enfermedad_gran_volumen', y='CD8', ax=axes[1], color='skyblue')
axes[1].set_title('CD8 según enfermedad')
axes[1].set_xlabel('Enfermedad')
axes[1].set_ylabel('CD8')

plt.tight_layout()
plt.show()

# Diagnóstico codificado vs enfermedad
dx_cols = ['Dx_BenignoAnal', 'Dx_BenignoOtro', 'Dx_Maligno', 'Dx_Verrugas', 'Dx_SinDato']
dx_corr = df_clean[dx_cols + ['Enfermedad_gran_volumen']].corr()['Enfermedad_gran_volumen']

plt.figure(figsize=(8, 5))
sns.barplot(x=dx_corr.values, y=dx_corr.index,palette='pastel')
plt.title('Diagnósticos codificados vs Enfermedad de Gran Volumen')
plt.xlabel('Correlación')
plt.tight_layout()
plt.show()

# Correlación general (mapa)

# Filtrar columnas numéricas y con al menos 80% de datos válidos
numeric_cols = df_clean.select_dtypes(include=['number', 'bool']).columns
filtered_numeric = df_clean[numeric_cols].dropna(axis=1, thresh=int(0.8 * len(df_clean)))

# Calcular matriz de correlación
corr_matrix = filtered_numeric.corr()

# Graficar con letra más pequeña para mayor legibilidad
plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 6})
plt.title("Mapa de Correlación entre Variables Numéricas", fontsize=14)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.show()






### imprimir solo unas variables
#variables = ['Enfermedad_gran_volumen', 'target_bin']

#df_reducido = df_clean[variables]

#print(df_reducido.head())



#REgresion lineal
print(40*'-')
print(df_clean[df_clean ['CD4'] > 40000])
print(40*'-')

df_clean = df_clean.drop([993, 999])
print(40*'-')
print(df_clean[df_clean ['CD4'] > 40000])
print(40*'-')

#df_clean['target_bin'] = df_clean['Enfermedad_gran_volumen'].apply(lambda x: 0 if x == 'No' else 1)
#df_clean.to_csv('./data/heart_disease_clean.csv', index=False)

# Entrenamiento del modelo de regresión lineal, no aplica para este caso
#train_regression_model(df_clean)

# Entrenamiento del modelo de regresión logística
train_logistic_model(df_clean)  

# Entrenamiento del modelo de Árbol de Decisión
train_decision_tree_model(df_clean)

# Entrenamiento del modelo de Máquina de soporte vectorial
train_svm_model(df_clean)

# Entrenamiento del modelo de Random Forest
train_random_forest_model(df_clean)

# Entrenamiento del modelo de Random Forest
#train_xgboost_model(df_clean)


## del colab
# Seleccionar variables numéricas y booleanas como predictores
X = df_clean.select_dtypes(include=['number', 'bool']).drop(columns=['Enfermedad_gran_volumen'])
y = df_clean['Enfermedad_gran_volumen'].astype(int)  # Asegurar que la variable objetivo sea numérica

# Imputar valores faltantes (media para numéricas)
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# División del conjunto en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Entrenar modelo Random Forest
model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train, y_train)

# Predicciones
y_pred = model_rf.predict(X_test)

# Evaluación
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=False)
accuracy = accuracy_score(y_test, y_pred)


print(conf_matrix)
print('\n')
print(report)
print('\n')
print(f"Accuracy: {accuracy:.4f}")

# Entrenar el modelo con class_weight='balanced' para mitigar el desbalance de clases
model_rf_balanced = RandomForestClassifier(random_state=42, class_weight='balanced')
model_rf_balanced.fit(X_train, y_train)

# Predicciones con modelo balanceado
y_pred_balanced = model_rf_balanced.predict(X_test)

# Evaluación
conf_matrix_bal = confusion_matrix(y_test, y_pred_balanced)
report_bal = classification_report(y_test, y_pred_balanced, output_dict=False)
accuracy_bal = accuracy_score(y_test, y_pred_balanced)


print(conf_matrix_bal)
print('\n')
print(report_bal)
print('\n')
print(f"Accuracy: {accuracy_bal:.4f}")



# Inicializar modelos
models = {
    "Random Forest": RandomForestClassifier(random_state=42, class_weight='balanced'),
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, class_weight='balanced'))
    ]),
    "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'))
    ])
}

# Evaluar cada modelo
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "Accuracy": acc,
        "F1-score (class 1)": report["1"]["f1-score"],
        "Precision (class 1)": report["1"]["precision"],
        "Recall (class 1)": report["1"]["recall"]
    }


# Reentrenar y graficar matrices de confusión para cada modelo
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axes[i], values_format='d', cmap='Blues')
    axes[i].set_title(f'Matriz de Confusión - {name}')

plt.tight_layout()
plt.show()


# Convertir resultados a DataFrame para visualización
results_df = pd.DataFrame(results).T
print("Comparación de Modelos")
print(results_df)



# Cargar curvas ROC
roc_log = pd.read_csv('./models/roc_logistic.csv')
roc_tree = pd.read_csv('./models/roc_arbol.csv')
roc_svm = pd.read_csv('./models/roc_svm.csv')
roc_rf = pd.read_csv('./models/roc_rf.csv')
#roc_xgboost = pd.read_csv('./models/roc_xgboost.csv')

# Cargar AUC
with open('./models/roc_logistic_auc.txt') as f:
    auc_log = f.read().strip().replace('AUC: ', '')

with open('./models/roc_arbol_auc.txt') as f:
    auc_tree = f.read().strip().replace('AUC: ', '')

with open('./models/roc_svm_auc.txt') as f:
    auc_svm = f.read().strip().replace('AUC: ', '')

with open('./models/roc_rf_auc.txt') as f:
    auc_rf = f.read().strip().replace('AUC: ', '')

#with open('./models/roc_xgboost_auc.txt') as f:
#    auc_rf = f.read().strip().replace('AUC: ', '')

# Graficar
plt.figure(figsize=(8, 6))
plt.plot(roc_log['fpr'], roc_log['tpr'], label=f'Regresión Logística (AUC = {auc_log})', color='blue')
plt.plot(roc_tree['fpr'], roc_tree['tpr'], label=f'Árbol de Decisión (AUC = {auc_tree})', color='green')
plt.plot(roc_svm['fpr'], roc_svm['tpr'], label=f'Máquina Soporte Vectorial (AUC = {auc_svm})', color='orange')
plt.plot(roc_rf['fpr'], roc_rf['tpr'], label=f'Random Forest (AUC = {auc_rf})', color='purple')
#plt.plot(roc_rf['fpr'], roc_rf['tpr'], label=f'XGBoost (AUC = {auc_rf})', color='purple')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')

plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('Curvas ROC - Comparación de Modelos')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 🔽 Guardar imagen en alta calidad
plt.savefig('./models/curvas_roc_comparadas.png', dpi=300)
plt.show()

show_and_save_confusion_matrices()