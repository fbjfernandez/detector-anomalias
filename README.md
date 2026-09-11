# 🛡️ Network Traffic Anomaly Detection (SecOps)

Un sistema integral de Machine Learning diseñado para clasificar tráfico de red y detectar anomalías en tiempo real, orientado a la prevención de brechas de seguridad en entornos corporativos y bancarios.

## 📌 Descripción del Proyecto
Este proyecto aborda un problema crítico en la ciberseguridad (Blue Team / SOC): la fatiga por alertas y los falsos positivos. Utilizando el dataset estándar de la industria **CIC-IDS-2017**, el sistema entrena modelos de Inteligencia Artificial para aprender el comportamiento normal de una red y clasificar ataques conocidos (como DDoS), reduciendo la intervención manual.

El proyecto destaca por su arquitectura *End-to-End*, que va desde el preprocesamiento de datos crudos hasta la visualización en un dashboard analítico interactivo.

## 🏗️ Arquitectura del Sistema
El proyecto está dividido en tres capas principales:

1. **Backend ML (Python):** Análisis exploratorio de datos (EDA), limpieza y entrenamiento de modelos (`Scikit-Learn`, `Pandas`, `NumPy`). Implementación de Random Forest (clasificación) e Isolation Forest (anomalías).
2. **API REST (FastAPI):** Un puente eficiente que sirve el modelo entrenado y procesa datos de tráfico simulado en tiempo real.
3. **Frontend Dashboard (React + Tailwind CSS):** Interfaz gráfica para analistas de seguridad, diseñada para visualizar el tráfico y resaltar las alertas de anomalías dinámicamente.

## 📊 Dataset Utilizado
* **Fuente:** CIC-IDS-2017 (Canadian Institute for Cybersecurity).
* **Características:** Contiene tráfico benigno y los ataques más comunes actualizados (Fuerza Bruta, DoS, Web Attacks, Infiltración).
* *Nota: Por motivos de peso, los archivos `.csv` originales no están incluidos en este repositorio.*
