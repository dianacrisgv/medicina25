# El problema es clasificar si un paciente tiene o no una enfermedad de gran volumen.
# Etiqueta: 'Enfermedad_gran_volumen' (target binaria: Sí / No)

import pandas as pd
from preprocess import clean_data
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.svm import SVC
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import ConfusionMatrixDisplay

from logistic_model import train_logistic_model
from decision_tree_model import train_decision_tree_model
from svm_model import train_svm_model
from random_forest_model import train_random_forest_model
from confusion_matrices import show_and_save_confusion_matrices
#from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score


# URL original del dataset
url = 'https://raw.githubusercontent.com/John624-web/curso-IA-John/refs/heads/main/IA%20Coloproctologia.csv'
df = pd.read_csv(url)

# Guardar una copia local
df.to_csv('./data/IAColoprocto_raw.csv', index=False)

# Cargar copia local
df = pd.read_csv('./data/IAColoprocto_raw.csv')

# Aplicar función de limpieza
df_clean = clean_data(df)

# Eliminar valores extremos o errores
print(40*'-')
print(df_clean[df_clean['CD4'] > 40000])
#df_clean = df_clean.drop([993, 999])
cd4_a_imputar = [993, 999]
df_clean.loc[cd4_a_imputar, 'CD4'] = df_clean['CD4'].mean()
print(40*'-')
print(df_clean[df_clean['CD8'] > 3000])
cd8_a_imputar = [271, 535, 562, 844]
df_clean.loc[cd8_a_imputar, 'CD8'] = df_clean['CD8'].mean()
print(40*'-')
print(df_clean[df_clean ['Edad'] > 120])
edad_a_imputar = [874, 876, 882, 887, 898, 974]
df_clean.loc[edad_a_imputar, 'Edad'] = df_clean['Edad'].mean()
print(40*'-')


# Codificar variable objetivo
df_clean['target_bin'] = df_clean['Enfermedad_gran_volumen'].apply(lambda x: 0 if x == 'No' else 1)

# Guardar versión limpia
df_clean.to_csv('./data/gran_volumen_clean.csv', index=False)

#Análisis Exploratorio de Datos (EDA)

# Distribución de clases
sns.countplot(data=df_clean, x='Enfermedad_gran_volumen', palette='pastel')
plt.title('Distribución de la Enfermedad de Gran Volumen')
plt.xlabel('Enfermedad (0 = No, 1 = Sí)')
plt.ylabel('Cantidad')
plt.savefig('./images/distribucionclases.png', dpi=300)
plt.show()

# Boxplot de edad por clase de enfermedad
sns.boxplot(data=df_clean, x='Enfermedad_gran_volumen', y='Edad', color='skyblue')
plt.title('Edad según presencia de Enfermedad de Gran Volumen')
plt.xlabel('Enfermedad (0 = No, 1 = Sí)')
plt.ylabel('Edad')
plt.savefig('./images/edadvsenfermedad.png', dpi=300)
plt.show()

# Boxplot de CD4 y CD8 por clase de enfermedad
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.boxplot(data=df_clean, x='Enfermedad_gran_volumen', y='CD4', ax=axes[0], color='skyblue')
axes[0].set_title('CD4 según enfermedad')
axes[0].set_xlabel('Enfermedad (0 = No, 1 = Sí)')
axes[0].set_ylabel('CD4')

sns.boxplot(data=df_clean, x='Enfermedad_gran_volumen', y='CD8', ax=axes[1], color='skyblue')
axes[1].set_title('CD8 según enfermedad')
axes[1].set_xlabel('Enfermedad (0 = No, 1 = Sí)')
axes[1].set_ylabel('CD8')

plt.tight_layout()
plt.savefig('./images/cd4cd8vsenfermedad.png', dpi=300)
plt.show()

# Diagnósticos vs enfermedad
dx_cols = ['Dx_BenignoAnal', 'Dx_BenignoOtro', 'Dx_Maligno', 'Dx_Verrugas', 'Dx_SinDato']
dx_corr = df_clean[dx_cols + ['Enfermedad_gran_volumen']].corr()['Enfermedad_gran_volumen']

sns.barplot(x=dx_corr.values, y=dx_corr.index, palette='pastel')
plt.title('Diagnósticos codificados vs Enfermedad')
plt.savefig('./images/dxcodificadosvsenfermedad.png', dpi=300)
plt.show()

# Mapa de correlación
# Filtrar columnas numéricas y con al menos 80% de datos válidos
numeric_cols = df_clean.select_dtypes(include=['number', 'bool']).columns
filtered_numeric = df_clean[numeric_cols].dropna(axis=1, thresh=int(0.8 * len(df_clean)))
# Calcular matriz de correlación
corr_matrix = filtered_numeric.corr()
#Graficar
plt.figure(figsize=(18, 14))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size": 6})
plt.title("Mapa de Correlación entre Variables Numéricas", fontsize=14)
plt.xticks(rotation=90, fontsize=6)
plt.yticks(fontsize=6)
plt.tight_layout()
plt.savefig('./images/mapadecorrelacion.png', dpi=300)
#plt.show()


#Preparación de datos para el modelado 
X = df_clean.select_dtypes(include=['number', 'bool']).drop(columns=['Enfermedad_gran_volumen'])
y = df_clean['Enfermedad_gran_volumen'].astype(int)

# Imputación de datos faltantes
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# División en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
# Lista para guardar métricas de todos los modelos
metricas_modelos = []

##Entranamiento modelos

# Entrenamiento del modelo de regresión logística
_, metricas_log =train_logistic_model(df_clean)  
metricas_modelos.append(metricas_log)

# Entrenamiento del modelo de Árbol de Decisión
_, metricas_tree =train_decision_tree_model(df_clean)
metricas_modelos.append(metricas_tree)

# Entrenamiento del modelo de Máquina de soporte vectorial
_, metricas_svm =train_svm_model(df_clean)
metricas_modelos.append(metricas_svm)

# Entrenamiento del modelo de Random Forest
_, metricas_rf = train_random_forest_model(df_clean)
metricas_modelos.append(metricas_rf)

# Convertir a DataFrame
df_metricas = pd.DataFrame(metricas_modelos)

# Guardar en CSV
df_metricas.to_csv('./models/comparacion_metricas_modelos.csv', index=False)

#Evaluación de modelos

#Visualización de métricas

# Mostrar en consola
print('\nComparación de métricas de modelos:')
print(df_metricas)


# Configuración del gráfico de comparación de métricas
ax = df_metricas.set_index('modelo').plot(kind='bar', figsize=(12, 6), colormap='Set2')

# Título y etiquetas
plt.title('Comparación de Métricas entre Modelos')
plt.ylabel('Valor')
plt.xlabel('Modelo')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Añadir etiquetas de valor a cada barra
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):  # Evita errores si hay valores NaN
        ax.annotate(f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=8, color='black', rotation=0)

plt.tight_layout()
plt.legend(loc='upper right')
plt.savefig('./models/comparacion_metricas.png', dpi=300)
plt.show()

#Evaluación con curvas ROC

roc_log = pd.read_csv('./models/roc_log.csv')
roc_tree = pd.read_csv('./models/roc_arbol.csv')
roc_svm = pd.read_csv('./models/roc_svm.csv')
roc_rf = pd.read_csv('./models/roc_rf.csv')

with open('./models/roc_logistic_auc.txt') as f:
    auc_log = f.read().strip().replace('AUC: ', '')
with open('./models/roc_arbol_auc.txt') as f:
    auc_tree = f.read().strip().replace('AUC: ', '')
with open('./models/roc_svm_auc.txt') as f:
    auc_svm = f.read().strip().replace('AUC: ', '')
with open('./models/roc_rf_auc.txt') as f:
    auc_rf = f.read().strip().replace('AUC: ', '')

# Graficar curvas ROC
plt.figure(figsize=(8, 8))
plt.plot(roc_log['fpr'], roc_log['tpr'], label=f'Regresión Logística (AUC = {auc_log})', color='blue')
plt.plot(roc_tree['fpr'], roc_tree['tpr'], label=f'Árbol de Decisión (AUC = {auc_tree})', color='green')
plt.plot(roc_svm['fpr'], roc_svm['tpr'], label=f'SVM (AUC = {auc_svm})', color='orange')
plt.plot(roc_rf['fpr'], roc_rf['tpr'], label=f'Random Forest (AUC = {auc_rf})', color='purple')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('Curvas ROC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('./models/curvas_roc_comparadas.png', dpi=300)
plt.show()

# Mostrar y guardar matrices de confusión finales

show_and_save_confusion_matrices()

