#Primera versión del main, cargar el csv
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Cargar el dataset
heart_disease = fetch_ucirepo(name='Heart Disease')

# Convertir a DataFrame
X = heart_disease.data.features
y = heart_disease.data.targets

# Unir ambos para una vista completa
df = pd.concat([X, y], axis=1)

# Guardar los datos como CSV
df.to_csv('./data/heart_disease_raw.csv', index=False)

# Mostrar las primeras filas
print(df.head())

""" #Segunda versión de main limpiando datos
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import clean_heart_data #importo la funcion de preprocess en main

# Cargar datos
df = pd.read_csv('./data/heart_disease_raw.csv')

# Limpiar datos
df_clean = clean_heart_data(df)

# Guardar versión limpia
df_clean.to_csv('./data/heart_disease_clean.csv', index=False)


print(df_clean.head()) 

# Visualizar la distribución de las variables numéricas
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

# Clasificación binaria: 0 = sin enfermedad, 1 = con enfermedad
df_clean['target_bin'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)

# Conteo de clases
sns.countplot(data=df_clean, x='target_bin', palette='pastel')
plt.title('Distribución de clases (ausencia vs presencia de enfermedad cardíaca)')
plt.xlabel('Enfermedad cardíaca (0 = No, 1 = Sí)')
plt.ylabel('Número de pacientes')
plt.show()



"""


#TErcera version main con regresion lineal

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import clean_heart_data
from regression_model import train_regression_model
from svm_model import train_svm_model

# Cargar datos
df = pd.read_csv('./data/heart_disease_raw.csv')

# Limpiar datos
df_clean = clean_heart_data(df)

# Crear variable binaria (0 = sin enfermedad, 1 = con enfermedad)
df_clean['target_bin'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)


# Guardar versión limpia
df_clean.to_csv('./data/heart_disease_clean.csv', index=False)

print(df_clean.head())


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

# Clasificación binaria: 0 = sin enfermedad, 1 = con enfermedad
df_clean['target_bin'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)

# Conteo de clases
sns.countplot(data=df_clean, x='target_bin', palette='pastel')
plt.title('Distribución de clases (ausencia vs presencia de enfermedad cardíaca)')
plt.xlabel('Enfermedad cardíaca (0 = No, 1 = Sí)')
plt.ylabel('Número de pacientes')
plt.show() """

""" # Entrenamiento de modelo de regresión
train_regression_model(df_clean)

df_clean['target_bin'] = df_clean['num'].apply(lambda x: 0 if x == 0 else 1)
df_clean.to_csv('./data/heart_disease_clean.csv', index=False) """



# Entrenamiento de modelo de regresión
train_svm_model(df_clean)