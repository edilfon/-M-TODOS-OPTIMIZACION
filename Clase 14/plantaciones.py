# Instalar librerías necesarias
!pip install openpyxl optuna

# Importar librerías
import pandas as pd
import optuna
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Subir el archivo
uploaded = files.upload()

# Obtener el nombre del archivo subido
file_name = next(iter(uploaded.keys()))

# Leer el archivo Excel correctamente
try:
    df = pd.read_excel(file_name, engine='openpyxl')  # Forzar el uso de openpyxl para evitar errores
    print("Archivo cargado correctamente:")
    print(df.head())  # Mostrar las primeras filas
except Exception as e:
    print(f"Error al leer el archivo: {e}")

# Verificar si hay valores nulos
print("\nValores nulos en el DataFrame:")
print(df.isnull().sum())

# Llenar valores nulos con la media de cada columna numérica
df.fillna(df.mean(numeric_only=True), inplace=True)

# Codificar variables categóricas si es necesario
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Dividir en variables predictoras (X) y variable objetivo (y)
X = df.drop(columns=['SUPERFICIE_PLANTACION'])  # Cambia por el nombre correcto de la columna objetivo
y = df['SUPERFICIE_PLANTACION']

# Separar en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir función de optimización con Optuna
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    error = mean_absolute_error(y_test, y_pred)
    
    return error

# Ejecutar la optimización
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Mostrar los mejores parámetros encontrados
print("\nMejores parámetros encontrados:")
print(study.best_params)

# Entrenar modelo final con los mejores parámetros
best_model = RandomForestRegressor(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)

# Evaluar el modelo
y_pred = best_model.predict(X_test)
error_final = mean_absolute_error(y_test, y_pred)

print("\nError final (MAE):", error_final)
