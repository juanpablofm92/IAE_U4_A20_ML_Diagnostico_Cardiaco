# Actividad 20: Diagnóstico Médico Cardiaco y Sensibilidad Clínica (Recall)

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 📌 1. Descripción del Proyecto

Este proyecto aborda la formulación rigurosa de un modelo predictivo para la detección temprana de cardiopatías isquémicas a partir de indicadores clínicos estandarizados (basados en el UCI Cleveland Heart Disease Dataset):
* Edad, presión arterial sistólica en reposo y colesterol sérico.
* Electrocardiograma en reposo (ECG).
* Frecuencia cardíaca máxima alcanzada y presencia de angina inducida por esfuerzo.
* Depresión del segmento ST inducida por ejercicio.

### Priorización de Métrica: Sensibilidad / Recall
En medicina diagnóstica, un **Falso Negativo (FN)** implica dar de alta erróneamente a un paciente en riesgo inminente de infarto, mientras que un **Falso Positivo (FP)** solo conduce a estudios confirmatorios complementarios. Por ello, el sistema calibra el umbral de decisión para maximizar el **Recall**:
$$\text{Recall} = \frac{TP}{TP + FN}$$

---

## 🔬 2. Modelos Evaluados

1. **Regresión Logística con Regularización L2 y ponderación balanceada**
2. **Bosques Aleatorios (Random Forest)**
3. **Gradient Boosting / XGBoost**

Se incluye la extracción analítica de la importancia de variables (`feature_importances_`) e interpretación fisiológica caso a caso (*Explainable AI*).

---

## 🚀 3. Instalación y Ejecución

```bash
pip install -r requirements.txt
python heart_diagnosis.py
```

---

## ⚖️ 4. Consideraciones Éticas en IA Médica

1. **No Maleficencia:** La calibración del punto de corte probabilístico debe favorecer la detección exhaustiva sobre la especificidad estricta para salvaguardar vidas humanas.
2. **Explicabilidad Clínica (XAI):** Rechazo al modelo de caja negra; el personal médico debe comprender el peso relativo de cada biomarcador en la decisión asistida.
3. **Equidad y Sesgo Demográfico:** Validación de desempeño desagregada por sexo y grupo etario para evitar infradiagnóstico sistemático.
