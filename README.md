# TOX-21: Predictor de Toxicidad Mitocondrial

[![DOI](https://zenodo.org/badge/1231808140.svg)](https://doi.org/10.5281/zenodo.20271209)

Este proyecto es una aplicación web desarrollada como parte de la asignatura de Bioinformática y Medicina de la Universidad da Coruña. Permite predecir la toxicidad mitocondrial de compuestos químicos mediante aprendizaje automático, analizando alteraciones en el potencial de membrana mitocondrial a partir de estructuras moleculares en formato SMILES.

## Acceso a la Aplicación

Puedes probar la aplicación en vivo a través del siguiente enlace: **[Despliegue en Vercel](https://tox21-predictor-two.vercel.app/)**

---

## Características Principales

- **Predicción de toxicidad**: Ingrese una estructura molecular en formato SMILES para obtener una predicción de toxicidad mitocondrial (0-1).
- **Visualización 2D interactiva**: Representación dinámica de la estructura molecular usando SmilesDrawer con temas personalizados.
- **Explicación SHAP**: Identifica los factores estructurales y fisicoquímicos que más influyen en la predicción mediante análisis de importancia.
- **Sistema de semáforo**: Resultados clasificados en tres categorías:
  - 🟢 Verde (≤35%): No tóxico / Estable
  - 🟡 Amarillo (35-65%): Advertencia / Zona gris
  - 🔴 Rojo (>65%): Tóxico / Crítico
- **Historial de consultas**: Almacena y permite revisar predicciones anteriores en el navegador mediante localStorage.
- **Ejemplos predefinidos**: Acceso rápido a compuestos no tóxicos (ej. Aspirina) y tóxicos (ej. Testosterona).

**Nota sobre el modelo utilizado**: El modelo entrenado en el servicio externo es el de oversampling. Los demás modelos (downsampling, reponderación, SMOTE) se han agrupado en la carpeta 'entrenamientos_probados' para mantener la información de qué se ha probado.

## Requisitos

- Navegador web moderno (Chrome, Firefox, Edge, etc.)
- Conexión a internet (para consumir la API de Hugging Face)

## Instalación y Uso

1. Accede a [https://tox21-predictor-two.vercel.app/](https://tox21-predictor-two.vercel.app/)
2. Ingrese la estructura molecular en formato SMILES en el campo de entrada.
3. Haga clic en "Predecir Toxicidad".
4. Observe el resultado con el sistema de semáforo y la explicación de factores SHAP.
5. Explore compuestos de ejemplo en la sección de ejemplos.

## Tecnologías Utilizadas

- **Frontend**: HTML5, CSS3, JavaScript (SmilesDrawer API)
- **API**: Hugging Face Inference API (paulahong-tox21-predictor-api.hf.space)
- **Almacenamiento**: localStorage para historial de consultas

## Estructura del Proyecto

```
tox21-predictor/
├── static/          # Archivos frontend (app.js, style.css, descriptors-glossary.js)
├── api/             # Scripts de la API
├── entrenamientos_probados/  # Modelos y scripts de entrenamiento (train_smote.py, etc.)
├── plots.zip        # Gráficos y visualizaciones generadas
├── index.html       # Interfaz principal
├── README.md        # Este documento
└── requirements.txt # Dependencias para entrenamiento (si aplica)
└── train_oversampling.py # Entrenamiento seleccionado para servicio externo

```

## Material Adicional y Presentación

- **Presentación para el examen**: **[Diapositivas en Canva](https://canva.link/xuistaowh0funtz)**

## Notas Importantes

- La aplicación depende de la API de Hugging Face, por lo que requiere conexión a internet.
- Los resultados pueden variar según el modelo entrenado en el servicio externo.
- El historial de consultas se guarda localmente en el navegador (localStorage) y se mantiene entre sesiones.
- La visualización 2D se genera dinámicamente usando SmilesDrawer para representar estructuras moleculares.
