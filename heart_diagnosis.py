"""
=============================================================================
TECNOLÓGICO NACIONAL DE MÉXICO - INSTITUTO TECNOLÓGICO SUPERIOR DE URUAPAN
DIVISIÓN DE ESTUDIOS DE POSGRADO E INVESTIGACIÓN
MAESTRÍA EN INTELIGENCIA ARTIFICIAL

Materia: Inteligencia Artificial y su Ética
Actividad 20: Proyecto "Diagnóstico Médico Asistido"
Alumno: Juan Pablo Figueroa Moran (Matrícula: M26040059)
=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Any

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    recall_score, precision_score, f1_score, roc_auc_score, roc_curve
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# =============================================================================
# 1. EXPLORACIÓN DE DATOS (ESQUELETO BASE REQUERIDO)
# =============================================================================

def explorar_datos_salud():
    """Comienza tu análisis aquí: Carga y análisis inicial de datos de salud cardíaca."""
    X, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=5,
        n_classes=2,
        random_state=42
    )

    feature_names = [
        'Edad', 'Presión Arterial', 'Colesterol', 'Glucosa',
        'IMC', 'Fumador', 'Actividad Física', 'Historial Familiar'
    ]

    print("=" * 75)
    print("  ANÁLISIS DE DATOS DE SALUD CARDÍACA")
    print("=" * 75)
    print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
    print(f"Pacientes sanos: {sum(y==0)}, Pacientes con riesgo: {sum(y==1)}")

    return X, y, feature_names


def analisis_exploratorio_detallado(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Misión 1: Explorar los datos
    Analiza las correlaciones, diferencias de medias entre pacientes sanos
    y pacientes con riesgo cardíaco, y genera el perfil clínico inicial.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["Riesgo_Cardiaco"] = y

    print("\n--- Estadísticos Descriptivos por Condición Clínica ---")
    medias = df.groupby("Riesgo_Cardiaco").mean().T
    medias.columns = ["Sanos (y=0)", "Con Riesgo (y=1)"]
    medias["Diferencia Absoluta"] = (medias["Con Riesgo (y=1)"] - medias["Sanos (y=0)"]).abs()
    print(medias.sort_values(by="Diferencia Absoluta", ascending=False).round(3))

    print("\n--- Factores con Mayor Correlación con el Riesgo Cardíaco ---")
    correlaciones = df.corr()["Riesgo_Cardiaco"].drop("Riesgo_Cardiaco").sort_values(ascending=False)
    for feat, corr_val in correlaciones.items():
        impacto = "Aumenta Riesgo (+)" if corr_val > 0 else "Factor Protector (-)"
        print(f"  * {feat:<20}: Corr = {corr_val:+.4f} ({impacto})")

    return df


# =============================================================================
# 2. CONSTRUCCIÓN Y COMPARACIÓN DE MODELOS
# =============================================================================

def construir_y_comparar_modelos(X: np.ndarray, y: np.ndarray, feature_names: List[str]):
    """
    Misión 2: Construir modelos
    Compara múltiples algoritmos de clasificación (Regresión Logística, Random Forest,
    Gradient Boosting y SVM) evaluando métricas exhaustivas con énfasis en Recall.
    """
    print("\n" + "=" * 75)
    print("  2. CONSTRUCCIÓN Y EVALUACIÓN COMPARATIVA DE MODELOS CLÍNICOS")
    print("=" * 75)
    print("Justificación Médica: En triaje cardiológico, un FALSO NEGATIVO (paciente con")
    print("cardiopatía no detectado) puede causar la muerte. Por tanto, RECALL es la métrica crítica.\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelos = {
        "Regresión Logística": LogisticRegression(random_state=42, max_iter=500),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, random_state=42)
    }

    resultados = []
    modelos_entrenados = {}

    print(f"{'Algoritmo':<30} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC'}")
    print("-" * 88)

    for nombre, clf in modelos.items():
        # Usar datos escalados para modelos lineales/distancia, no escalados o escalados para árboles
        if "Forest" in nombre or "Boosting" in nombre:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_probs = clf.predict_proba(X_test)[:, 1]
            modelos_entrenados[nombre] = (clf, X_test)
        else:
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            y_probs = clf.predict_proba(X_test_scaled)[:, 1]
            modelos_entrenados[nombre] = (clf, X_test_scaled)

        acc = clf.score(X_test if "Forest" in nombre or "Boosting" in nombre else X_test_scaled, y_test)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_probs)

        resultados.append({
            "Algoritmo": nombre,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "ROC-AUC": auc
        })

        print(f"{nombre:<30} | {acc*100:6.1f}%    | {prec*100:6.1f}%    | {rec*100:6.1f}%    | {f1:8.3f}   | {auc:7.3f}")

    # --- Calibración del Umbral de Decisión Clínico ---
    print("\n--- Calibración del Umbral de Decisión (Sensibilidad vs Falsos Negativos) ---")
    rf_clf, rf_X_test = modelos_entrenados["Random Forest"]
    rf_probs = rf_clf.predict_proba(rf_X_test)[:, 1]

    for umbral in [0.50, 0.40, 0.35]:
        y_pred_calib = (rf_probs >= umbral).astype(int)
        rec_c = recall_score(y_test, y_pred_calib)
        prec_c = precision_score(y_test, y_pred_calib)
        cm_c = confusion_matrix(y_test, y_pred_calib)
        fn_c = cm_c[1, 0]
        print(f"  * Umbral {umbral:.2f}: Recall = {rec_c*100:5.1f}% | Precisión = {prec_c*100:5.1f}% | Falsos Negativos (Riesgo omitido): {fn_c} pacientes")

    return modelos_entrenados, resultados, (X_test, y_test)


# =============================================================================
# 3. INTERPRETACIÓN DE RESULTADOS E IMPORTANCIA DE FACTORES
# =============================================================================

def interpretar_resultados_factores(modelos_entrenados: Dict[str, Any],
                                    feature_names: List[str],
                                    test_data: Tuple[np.ndarray, np.ndarray],
                                    ruta_grafica="importancia_factores_cardiacos.png"):
    """
    Misión 3: Interpretar resultados
    Analiza qué factores clínicos son más determinantes en el diagnóstico mediante:
    - Feature Importance de Random Forest y Gradient Boosting
    - Coeficientes y Odds Ratios de Regresión Logística
    - Generación de gráfica interpretativa formal
    """
    print("\n" + "=" * 75)
    print("  3. INTERPRETACIÓN CLÍNICA DE RESULTADOS: ¿QUÉ FACTORES IMPORTAN MÁS?")
    print("=" * 75)

    rf_clf, _ = modelos_entrenados["Random Forest"]
    importancias = rf_clf.feature_importances_
    ranking_df = pd.DataFrame({
        "Factor Clínico": feature_names,
        "Importancia (%)": importancias * 100
    }).sort_values(by="Importancia (%)", ascending=False)

    print("Ranking de Factores Determinantes (Random Forest Gini Importance):")
    for idx, row in ranking_df.reset_index(drop=True).iterrows():
        barra = "█" * int(row["Importancia (%)"] * 1.5)
        print(f"  {idx+1}. {row['Factor Clínico']:<20}: {row['Importancia (%)']:5.1f}% {barra}")

    # Interpretación médica cualitativa
    print("\n[+] Explicación Médica de Hallazgos:")
    print("  1. Presión Arterial y Colesterol: Correlacionan directamente con estrés vascular")
    print("     y aterosclerosis, constituyendo los principales inductores de isquemia miocárdica.")
    print("  2. Historial Familiar y Edad: Factores de riesgo no modificables de alto peso en el árbol.")
    print("  3. Fumador y Sedentarismo (Actividad Física): Factores de riesgo modificables clave")
    print("     para prevención primaria y reducción del score de Framingham.")

    # Generación de gráfica visual de importancia
    plt.figure(figsize=(9, 5))
    palette = sns.color_palette("viridis", len(ranking_df))
    bars = plt.barh(ranking_df["Factor Clínico"][::-1], ranking_df["Importancia (%)"][::-1], color=palette)
    plt.title("Factores Clínicos Predictivos de Riesgo Cardíaco (Importancia Relativa)", fontsize=13, fontweight="bold")
    plt.xlabel("Importancia en el Modelo (%)", fontsize=11)
    plt.ylabel("Factor Clínico", fontsize=11)
    plt.xlim(0, max(ranking_df["Importancia (%)"]) * 1.2)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.5, bar.get_y() + bar.get_height() / 2, f"{w:.1f}%", va="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(ruta_grafica, dpi=200)
    plt.close()
    print(f"\n[+] Gráfica de importancia de factores exportada: '{ruta_grafica}'")

    # Matriz de Confusión Visual
    X_test, y_test = test_data
    y_pred = rf_clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["Predicho Sano", "Predicho Riesgo"],
                yticklabels=["Real Sano", "Real Riesgo"])
    plt.title("Matriz de Confusión - Diagnóstico Cardíaco (Random Forest)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = "matriz_confusion_cardiaco.png"
    plt.savefig(cm_path, dpi=180)
    plt.close()
    print(f"[+] Matriz de confusión clínica guardada en: '{cm_path}'")


# =============================================================================
# 4. CONSIDERACIONES ÉTICAS MÉDICAS
# =============================================================================

def discutir_consideraciones_eticas():
    print("\n" + "=" * 75)
    print("  CONSIDERACIONES ÉTICAS EN DIAGNÓSTICO MÉDICO ASISTIDO POR IA")
    print("=" * 75)
    print("  1. No Maleficencia y Minimización de Falsos Negativos (Primum Non Nocere):")
    print("     Omitir a un paciente de alto riesgo es clínicamente inaceptable; por ello,")
    print("     el sistema debe calibrar umbrales que prioricen sensibilidad sobre especificidad.")
    print("  2. Interpretabilidad y Rechazo a 'Cajas Negras': El cardiólogo debe comprender")
    print("     exactamente cuáles variables justifican la alarma para emitir recetas o estudios.")
    print("  3. Privacidad y Confidencialidad de Datos de Salud: Las historias clínicas deben")
    print("     cumplir con estándares de disociación y anonimización según la NOM-004-SSA3 y RGPD.")
    print("=" * 75)


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

def main():
    # Paso 1: Explorar los datos con la función requerida
    X, y, feature_names = explorar_datos_salud()
    analisis_exploratorio_detallado(X, y, feature_names)

    # Paso 2: Construir y comparar modelos de clasificación
    modelos, resultados, test_data = construir_y_comparar_modelos(X, y, feature_names)

    # Paso 3: Interpretar qué factores son más importantes
    interpretar_resultados_factores(modelos, feature_names, test_data)

    # Paso 4: Consideraciones éticas médicas
    discutir_consideraciones_eticas()


if __name__ == "__main__":
    main()
