import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib # Este módulo se utiliza para guardar el modelo entrenado

def train_regression_model(df, target='thalach'):
    # Definir columnas a eliminar dinámicamente
    drop_candidates = [target, 'num', 'target_bin']
    cols_to_drop = [col for col in drop_candidates if col in df.columns]

    # Variables predictoras
    features = df.drop(columns=cols_to_drop).columns.tolist()
    X = df[features]
    y = df[target]

    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(model.coef_)
    print(model.intercept_)
    intercepto = model.intercept_
    coef = model.coef_

    #Mostrar ecuación del modelo
    ecuacion = f"y = {intercepto:.2f}"
    for i, c in zip(X.columns, coef):
        ecuacion += f" + ({c:.2f})*{i}"
    print(ecuacion)

    # Predicciones
    y_pred = model.predict(X_test)

    # Guardar el modelo
    joblib.dump(model, './models/regresion_lineal_thalach.pkl')

    # Evaluación
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'\n📊 Evaluación del modelo de regresión lineal:')
    print(f'MSE: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')

    # Visualización
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
    plt.xlabel('Valor Real')
    plt.ylabel('Valor Predicho')
    plt.title(f'Regresión Lineal sobre {target}')
    plt.tight_layout()
    plt.show()