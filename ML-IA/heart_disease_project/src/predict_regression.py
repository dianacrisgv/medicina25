import pandas as pd
import joblib

# Cargar el modelo entrenado
model = joblib.load('./models/regresion_lineal_thalach.pkl')

# Crear nuevo paciente con las mismas variables predictoras usadas en entrenamiento
nuevo_paciente = pd.DataFrame([{
    'age': 60, 'sex': 1, 'cp': 3, 'trestbps': 140, 'chol': 220,
    'fbs': 0, 'restecg': 1, 'exang': 0, 'oldpeak': 1.0,
    'slope': 2, 'ca': 0.0, 'thal': 3.0
}])

# Asegúrate de que las columnas estén en el orden correcto
# Puedes guardar el orden de X.columns en un archivo o definirlo aquí manualmente
column_order = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
nuevo_paciente = nuevo_paciente[column_order]

# Realizar la predicción
prediccion = model.predict(nuevo_paciente)

# Imprimir Informe de Predicción
print('\n\nPREDICCION')
print(f'\tPredicción: Frecuencia cardíaca máxima esperada = {prediccion[0]:.2f}')
print(f'_'*70)