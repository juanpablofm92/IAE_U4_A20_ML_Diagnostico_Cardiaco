# Actividad 20: Proyecto "Diagnóstico Médico Asistido"

**Tecnológico Nacional de México**  
**Instituto Tecnológico Superior de Uruapan**  
**División de Estudios de Posgrado e Investigación**  
**Maestría en Inteligencia Artificial**  

* **Asignatura:** Inteligencia Artificial y su Ética (Unidad 4)  
* **Alumno:** Juan Pablo Figueroa Moran  
* **Matrícula:** M26040059  

---

## 🎯 Contexto y Misión

Como parte de un equipo de investigación médica, desarrollamos un sistema inteligente para predecir cardiopatías en pacientes a partir de exámenes clínicos de rutina. El proyecto cumple rigurosamente con tres objetivos cardinales:

1. **Explorar los datos:** Análisis de historiales clínicos anonimizados (1,000 pacientes, 8 variables fisiológicas) identificando diferencias de medias y factores correlacionados con el riesgo cardíaco.
2. **Construir modelos:** Implementación y comparativa de múltiples algoritmos supervisados:
   * Regresión Logística (modelo base probabilístico e interpretable).
   * Support Vector Machine (clasificador de margen máximo con kernel RBF).
   * Random Forest (ensamble de 150 árboles de decisión con profundidad controlada).
   * Gradient Boosting (ensamble secuencial con minimización de gradiente).
   * **Priorización de Sensibilidad (*Recall*):** Calibración del umbral de decisión ($\tau = 0.35 - 0.40$) para reducir los falsos negativos a menos de 2-4 pacientes en triaje clínico.
3. **Interpretar resultados:** Determinación cuantitativa de los factores clínicos determinantes mediante importancia de Gini y su justificación médica para soporte a la toma de decisiones del médico especialista.

---

## 📋 Variables Clínicas del Estudio

* `Edad`: Rango etario del paciente.
* `Presión Arterial`: Tensión arterial sistólica/diastólica.
* `Colesterol`: Colesterol sérico total en sangre.
* `Glucosa`: Nivel glucémico en ayunas.
* `IMC`: Índice de masa corporal.
* `Fumador`: Tabaquismo activo / pasivo.
* `Actividad Física`: Nivel de ejercicio regular reportado.
* `Historial Familiar`: Antecedentes genéticos de cardiopatía temprana.

---

## 📂 Estructura del Repositorio

```text
IAE_U4_A20_ML_Diagnostico_Cardiaco/
├── heart_diagnosis.py                # Pipeline integral: Exploración, Modelado e Interpretación
├── importancia_factores_cardiacos.png# Gráfica de barras de factores clínicos determinantes
├── matriz_confusion_cardiaco.png     # Matriz de confusión clínica con umbral calibrado
├── requirements.txt                  # Dependencias del proyecto (scikit-learn, pandas, seaborn)
└── README.md                         # Documentación técnica completa
```

---

## 🚀 Instrucciones de Ejecución

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar pipeline completo
python heart_diagnosis.py
```

---

## ⚖️ Consideraciones Éticas en IA Médica

1. **Principio de No Maleficencia (*Primum Non Nocere*):** En diagnóstico preventivo, un falso negativo omite el tratamiento oportuno de una afección potencialmente letal. Por ello, se ajustan los umbrales de probabilidad para maximizar la sensibilidad diagnóstica.
2. **Explicabilidad y Cajas Negras:** Ninguna decisión terapéutica o intervención quirúrgica debe basarse en un algoritmo opaco; los biomarcadores de mayor peso deben ser transparentes e interpretables para el médico tratante.
3. **Privacidad de Expedientes Clínicos:** Se respeta la confidencialidad estricta y anonimización de identificadores directos conforme a la NOM-004-SSA3 del Expediente Clínico y la LFPDPPP.
