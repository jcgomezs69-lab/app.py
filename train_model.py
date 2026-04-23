# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# 1. Cargar datos del estudio de adopción
# Variables típicas: edad, educación, ingresos, percepción_utilidad, 
# facilidad_uso, soporte_social, actitud_hacia_tech, adopcion (target)
df = pd.read_csv('data/adopcion_tecnologia.csv')

# 2. Preprocesamiento
# Codificar variables categóricas
le = LabelEncoder()
categorical_cols = ['educacion', 'genero', 'sector_laboral']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Definir features y target
X = df.drop('adopcion', axis=1)  # adopcion: 0=No adopta, 1=Adopta
y = df['adopcion']

# 3. División train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Entrenar Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    class_weight='balanced'  # Útil si hay desbalance de clases
)
rf_model.fit(X_train, y_train)

# 5. Evaluación
y_pred = rf_model.predict(X_test)
print("Accuracy:", rf_model.score(X_test, y_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Guardar modelo y metadatos
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("✅ Modelo guardado exitosamente")
