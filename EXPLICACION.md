# Explicación del Proyecto: Detección de Fraude con K-Means

Este documento detalla el funcionamiento interno de la aplicación para que puedas explicarlo con seguridad.

## 1. ¿Qué es y qué hace?
Es una aplicación web construida con **Django** (Python) que aplica el algoritmo de Machine Learning **K-Means** para agrupar transacciones bancarias y detectar patrones (posibles fraudes) en tiempo real.

En lugar de usar imágenes pre-generadas, la aplicación permite al usuario definir dinámicamente cuántos grupos (**clusters**) quiere buscar en los datos.

## 2. Flujo de Funcionamiento (Paso a Paso)

1.  **Entrada del Usuario**: El usuario ingresa un número de clusters (por defecto 6) en la página web y presiona "Actualizar".
2.  **Procesamiento en el Servidor (`views.py`)**:
    *   Django recibe el número a través de una petición `GET`.
    *   Carga el dataset `creditcard.csv` (optimizando memoria al cargar solo las columnas `V10` y `V14`, que son las más relevantes).
    *   **Ejecución de K-Means**: Entrena el algoritmo `KMeans` de la librería `scikit-learn` con los datos cargados y el número de clusters solicitado.
3.  **Generación de Gráficos**:
    *   Usa `matplotlib` para generar dos gráficos:
        *   **Dispersión**: Muestra los puntos de datos y sus centros (centroides).
        *   **Fronteras de Decisión**: Muestra las regiones que pertenecen a cada cluster (usando un clasificador KNN auxiliar para "pintar" el fondo).
    *   **Conversión a Imagen**: En lugar de guardar un archivo `.png` en el disco, convierte la gráfica a **Base64** (texto) en memoria.
4.  **Renderizado (`index.html`)**:
    *   Django envía las cadenas de texto Base64 a la plantilla HTML.
    *   El navegador interpreta ese texto como una imagen y la muestra al instante.

## 3. Estructura de Archivos (Templates)

La carpeta `templates` contiene la interfaz gráfica. En este proyecto solo tenemos un archivo principal:

### `api/templates/api/index.html`
Este archivo es la "cara" de la aplicación. No es solo HTML estático, utiliza el motor de plantillas de Django (**Jinja2**) para mostrar datos dinámicos.

**Partes Clave:**
*   **Formulario (`<form>`)**: Donde está el `input` numérico. Envía el dato al servidor cuando pulsas el botón.
*   **CSS "Glassmorphism"**: En la etiqueta `<style>`, definimos un diseño moderno con fondos oscuros semitransparentes (`backdrop-filter: blur`).
*   **Variables de Django (`{{ }}`)**:
    *   `{{ n_clusters }}`: Muestra el número actual seleccionado.
    *   `{{ counts }}`: Se usa en un bucle (`{% for %}`) para llenar la tabla de estadísticas automáticamente.
    *   `{{ scatter_plot }}`: Aquí se inyecta la imagen generada en Python directamente en el atributo `src` de la etiqueta `<img>`.

## 4. Preguntas Probables del Profesor

*   **P: ¿Por qué usas V10 y V14?**
    *   **R:** Basado en el análisis exploratorio del Notebook original, estas dos características mostraron la mejor separación visual entre transacciones normales y fraudulentas.
*   **P: ¿El modelo se re-entrena cada vez?**
    *   **R:** En esta versión web, sí. Al cambiar el número de clusters, el algoritmo `fit_predict` corre de nuevo para mostrar los nuevos centroides. Para producción masiva se usaría un modelo pre-entrenado, pero aquí queremos interactividad educativa.
*   **P: ¿Qué es el archivo `generate_kmeans_6.py`?**
    *   **R:** Era un script auxiliar para generar assets estáticos. Ahora el funcionamiento es dinámico desde `views.py`.

---
*DeepMind Antigravity - Generado para apoyo académico.*
