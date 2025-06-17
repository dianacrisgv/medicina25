import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from preprocess import clean_heart_data


""" # URL del dataset
url = 'https://raw.githubusercontent.com/John624-web/curso-IA-John/refs/heads/main/IA%20Coloproctologia.csv'

# Cargar datos
df = pd.read_csv(url)

print(df.head())
df.head()

# Guardar los datos como CSV
df.to_csv('./data/IAColoprocto_raw.csv', index=False) """


##
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import clean_data
from regression_model import train_regression_model
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

# Guardar los datos como CSV
df.to_csv('./data/IAColoprocto_raw.csv', index=False)

# Cargar datos
df = pd.read_csv('./data/IAColoprocto_raw.csv')

# Limpiar datos
#df_clean = clean_data(df)

df_clean = df

# Guardar versión limpia
df_clean.to_csv('./data/IAColoprocto_clean.csv', index=False) 

"""
#Estadísticas descriptivas y tipos de variables raw
print("Primeras filas del dataset:") 
print(df.head())

print("\nResumen estadístico:") 
print(df.describe())

print("\nTipos de datos:") 
print(df.dtypes)

#Estadísticas descriptivas y tipos de variables clean
print("Primeras filas del dataset limpio:") 
print(df_clean.head())

print("\nResumen estadístico:") 
print(df_clean.describe())

print("\nTipos de datos:") 
print(df_clean.dtypes) """

""" # Visualizar la distribución de las variables numéricas
df_clean.hist(bins=20, figsize=(14, 10), color='skyblue')
plt.suptitle('Distribuciones de variables numéricas')
plt.tight_layout()
plt.show()

# Visualizar las Matriz de correlaciones
plt.figure(figsize=(12, 10))
correlation = df_clean.corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Matriz de correlaciones')
plt.show()
 """
# Clasificación binaria: 0 = sin enfermedad, 1 = con enfermedad
df_clean['target_bin'] = df_clean['Enfermedad_gran_volumen'].apply(lambda x: 0 if x == 'No' else 1)

print(df_clean.head())

print(df_clean.describe())



# Conteo de clases
sns.countplot(data=df_clean, x='target_bin', palette='pastel')
plt.title('Distribución de clases (ausencia vs presencia de enfermedad gran volumen)')
plt.xlabel('Enfermedad gran volumen (0 = No, 1 = Sí)')
plt.ylabel('Número de pacientes')
plt.show()

### imprimir solo unas variables
variables = ['Enfermedad_gran_volumen', 'target_bin']

df_reducido = df_clean[variables]

print(df_reducido.head())

#REgresion lineal



df_clean['target_bin'] = df_clean['Enfermedad_gran_volumen'].apply(lambda x: 0 if x == 'No' else 1)
df_clean.to_csv('./data/heart_disease_clean.csv', index=False)

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