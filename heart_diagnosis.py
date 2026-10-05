"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 20: Machine Learning - Diagnóstico Médico Cardiaco y Sensibilidad (Recall)
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple, Dict

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, roc_auc_score


def generar_dataset_cardiaco_uci(n_muestras: int = 300, random_seed: int = 42) -> pd.DataFrame:
    """
    Genera un conjunto de datos sintético rigurosamente modelado a partir de la
    distribución multivariada del UCI Cleveland Heart Disease Dataset:
    - age: edad en años
    - trestbps: presión arterial sistólica en reposo (mm Hg)
    - chol: colesterol sérico (mg/dl)
    - restecg: resultados electrocardiográficos (0: normal, 1: anomalía ST-T, 2: hipertrofia)
    - thalach: frecuencia cardíaca máxima alcanzada
    - exang: angina inducida por ejercicio (1: sí, 0: no)
    - oldpeak: depresión del ST inducida por ejercicio en relación al reposo
    - target: presencia de cardiopatía (1: enfermo, 0: sano)
    """
    np.random.seed(random_seed)
    
    age = np.random.normal(54.4, 9.0, n_muestras).clip(29, 77).round().astype(int)
    trestbps = np.random.normal(131.6, 17.5, n_muestras).clip(94, 200).round().astype(int)
    chol = np.random.normal(246.0, 51.8, n_muestras).clip(126, 564).round().astype(int)
    restecg = np.random.choice([0, 1, 2], size=n_muestras, p=[0.48, 0.50, 0.02])
    thalach = (220 - age * 0.7 + np.random.normal(0, 15, n_muestras)).clip(71, 202).round().astype(int)
    exang = np.random.choice([0, 1], size=n_muestras, p=[0.67, 0.33])
    oldpeak = np.random.exponential(1.0, n_muestras).clip(0.0, 6.2).round(1)

    # Lógica de riesgo clínico probabilístico basada en literatura cardiológica
    score_riesgo = (
        0.04 * (age - 50) +
        0.02 * (trestbps - 130) +
        0.01 * (chol - 240) +
        0.60 * exang +
        0.50 * oldpeak -
        0.02 * (thalach - 150) +
        0.30 * restecg
    )
    prob_enfermedad = 1.0 / (1.0 + np.exp(-score_riesgo))
    target = (np.random.rand(n_muestras) < prob_enfermedad).astype(int)

    df = pd.DataFrame({
        "edad": age,
        "presion_arterial": trestbps,
        "colesterol": chol,
        "ecg_reposo": restecg,
        "frec_cardiaca_max": thalach,
        "angina_ejercicio": exang,
        "depresion_st": oldpeak,
        "cardiopatia": target
    })
    return df


def realizar_analisis_exploratorio(df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("  1. ANÁLISIS EXPLORATORIO DE DATOS DE SALUD (EDA)")
    print("=" * 75)
    print(f"Dimensiones del dataset: {df.shape[0]} pacientes, {df.shape[1]} variables.")
    print("\nDistribución de la variable objetivo (Cardiopatía):")
    conteo = df["cardiopatia"].value_counts()
    print(f"  - Sanos (0):      {conteo.get(0, 0)} ({conteo.get(0, 0)/len(df)*100:.1f}%)")
    print(f"  - Cardiópatas (1): {conteo.get(1, 0)} ({conteo.get(1, 0)/len(df)*100:.1f}%)")

    print("\nEstadísticos descriptivos agrupados por condición clínica:")
    resumen = df.groupby("cardiopatia")[["edad", "presion_arterial", "colesterol", "frec_cardiaca_max", "depresion_st"]].mean()
    print(resumen.round(2))


def entrenar_y_evaluar_modelos(df: pd.DataFrame):
    print("\n" + "=" * 75)
    print("  2. MODELADO COMPARATIVO PRIORIZANDO RECALL (SENSIBILIDAD)")
    print("=" * 75)
    print("Justificación Médica: En el triaje diagnóstico, un Falso Negativo (paciente enfermo no detectado)")
    print("puede resultar mortal. Por ende, la métrica crítica a maximizar es el RECALL.\n")

    X = df.drop(columns=["cardiopatia"])
    y = df["cardiopatia"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Intentar importar XGBoost, si no usar GradientBoostingClassifier
    try:
        from xgboost import XGBClassifier
        xgb_model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, eval_metric="logloss")
        nombre_xgb = "XGBoost Classifier"
    except ImportError:
        xgb_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        nombre_xgb = "Gradient Boosting (XGBoost Fallback)"

    modelos = {
        "Regresión Logística (L2)": (LogisticRegression(class_weight="balanced", random_state=42), True),
        "Random Forest": (RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42), False),
        nombre_xgb: (xgb_model, False)
    }

    mejor_modelo = None
    mejor_recall = -1.0
    nombre_mejor_modelo = ""

    for nombre, (clf, usar_escalado) in modelos.items():
        X_tr = X_train_scaled if usar_escalado else X_train
        X_te = X_test_scaled if usar_escalado else X_test

        clf.fit(X_tr, y_train)
        
        # Inferencia con ajuste de umbral para optimizar Recall (umbral 0.4 en lugar de 0.5)
        probs = clf.predict_proba(X_te)[:, 1]
        y_pred = (probs >= 0.40).astype(int)

        rec = recall_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        auc = roc_auc_score(y_test, probs)

        print(f">>> {nombre}")
        print(f"    Recall (Sensibilidad): {rec*100:.2f}% | Precisión: {prec*100:.2f}% | ROC-AUC: {auc:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        print(f"    Matriz Confusión [TN: {cm[0,0]}, FP: {cm[0,1]} | FN: {cm[1,0]}, TP: {cm[1,1]}]")
        print()

        if rec > mejor_recall:
            mejor_recall = rec
            mejor_modelo = clf
            nombre_mejor_modelo = nombre

    # 3. Importancia de características
    analizar_importancia_caracteristicas(mejor_modelo, X.columns.tolist(), nombre_mejor_modelo)
    
    # 4. Interpretación clínica de un caso
    interpretar_caso_clinico(mejor_modelo, scaler, X.columns.tolist(), nombre_mejor_modelo)


def analizar_importancia_caracteristicas(modelo, feature_names, nombre_modelo):
    print("=" * 75)
    print(f"  3. IMPORTANCIA DE CARACTERÍSTICAS ({nombre_modelo})")
    print("=" * 75)

    if hasattr(modelo, "feature_importances_"):
        importancias = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        importancias = np.abs(modelo.coef_[0])
    else:
        return

    ranking = sorted(zip(feature_names, importancias), key=lambda x: x[1], reverse=True)
    for feat, imp in ranking:
        barra = "█" * int(imp * 40 / max(importancias))
        print(f"  {feat:<20} : {imp:.4f} {barra}")


def interpretar_caso_clinico(modelo, scaler, feature_names, nombre_modelo):
    print("\n" + "=" * 75)
    print("  4. INTERPRETACIÓN CLÍNICA DE UNA PREDICCIÓN INDIVIDUAL")
    print("=" * 75)
    
    # Caso de paciente simulado
    paciente = pd.DataFrame([{
        "edad": 62,
        "presion_arterial": 158,
        "colesterol": 286,
        "ecg_reposo": 1,
        "frec_cardiaca_max": 110,
        "angina_ejercicio": 1,
        "depresion_st": 2.8
    }])

    print("Datos del Paciente ingresado:")
    for col, val in paciente.iloc[0].items():
        print(f"  * {col}: {val}")

    if "Regresión" in nombre_modelo:
        X_p = scaler.transform(paciente)
    else:
        X_p = paciente

    prob = modelo.predict_proba(X_p)[0, 1]
    diagnostico = "Alto Riesgo Cardiopático" if prob >= 0.40 else "Bajo Riesgo Clínico"

    print(f"\n[+] Probabilidad estimada de cardiopatía: {prob*100:.1f}%")
    print(f"[+] Dictamen Asistido por IA: {diagnostico}")
    print("\nRazonamiento Clínico:")
    print("  - Factores agravantes: Depresión del segmento ST elevada (2.8 mm) y angina durante ejercicio.")
    print("  - Presión arterial sistólica elevada (158 mm Hg) junto a frecuencia cardíaca máxima disminuida.")
    print("  - Recomendación: Canalización prioritaria a ecocardiograma de esfuerzo y cateterismo electivo.")

    print("\n" + "=" * 75)
    print("  CONSIDERACIONES ÉTICAS EN SALUD Y DIAGNÓSTICO MÉDICO ASISTIDO POR IA")
    print("=" * 75)
    print("""
    1. Responsabilidad y Autonomía Médica: El algoritmo es una herramienta de soporte
       a la decisión; la responsabilidad clínica final recae ineludiblemente en el médico especialista.
    2. Equidad y No Discriminación: Evitar sesgos por edad o género (e.g. subdiagnóstico
       habitual de infartos femeninos por presentación atípica de síntomas).
    3. Explicabilidad Algorítmica (XAI): En medicina no es aceptable una 'caja negra';
       debe garantizarse que las variables que conducen al dictamen sean auditables.
    """)


def main():
    print("=" * 75)
    print("  TECNM / ITSU - DIAGNÓSTICO CARDIOPÁTICO CON MACHINE LEARNING")
    print("=" * 75)
    df = generar_dataset_cardiaco_uci(n_muestras=320, random_seed=42)
    realizar_analisis_exploratorio(df)
    entrenar_y_evaluar_modelos(df)


if __name__ == "__main__":
    main()
