# Guía de Despliegue - Sistema de Detección (Tumor y Emociones)

Esta guía detalla los pasos necesarios para ejecutar y desplegar la aplicación Django que incluye el sistema de Detección de Tumores y Análisis de Emociones.

## Requisitos Previos

- Sistema Operativo: Linux (Recomendado)
- Python 3.8+
- Bibliotecas necesarias (listadas en los imports, se recomienda un entorno virtual)

El proyecto ya cuenta con un entorno virtual configurado en `.ply`.

## Pasos para Ejecutar (Entorno Local/Desarrollo)

### 1. Activar el Entorno Virtual

Es fundamental usar el entorno virtual `.ply` que contiene las dependencias instaladas (Django, Pillow, joblib, scikit-learn, etc).

```bash
source .ply/bin/activate
```

### 2. Entrenar el Modelo de Detección de Tumores

Si no se ha generado el archivo `tumor_model.joblib` o se desea reentrenar:

```bash
python scripts/train_tumor_model.py
```
*Este paso generará el archivo `tumor_model.joblib` en la raíz del proyecto.*

### 3. Verificar Archivos Estáticos y Migraciones

Asegúrese de que la base de datos y los archivos estáticos estén listos:

```bash
python manage.py migrate
python manage.py collectstatic --noinput
```

### 4. Iniciar el Servidor de Desarrollo

Para ejecutar la aplicación localmente en el puerto 8000:

```bash
python manage.py runserver 0.0.0.0:8000
```
*Ahora puede acceder a la aplicación en `http://localhost:8000/` o desde su navegador.*

---

## Despliegue en Producción (Básico)

Para un entorno más robusto (ej. servidor VPS), no se debe usar `runserver`. Se recomienda usar **Gunicorn**.

### 1. Instalar Gunicorn (si no está instalado)

```bash
pip install gunicorn
```

### 2. Ejecutar con Gunicorn

Ejecute la aplicación utilizando Gunicorn como servidor WSGI:

```bash
gunicorn fraud_detection.wsgi:application --bind 0.0.0.0:8000
```

### 3. Notas de Configuración (`settings.py`)

Antes de pasar a producción real, asegúrese de editar `fraud_detection/settings.py`:

- **DEBUG**: Cambiar a `False`.
  ```python
  DEBUG = False
  ```
- **ALLOWED_HOSTS**: Añadir la IP o dominio del servidor.
  ```python
  ALLOWED_HOSTS = ['midominio.com', '192.168.1.100']
  ```

---

## Estructura de URLs

- `/`: **Detección de Tumores** (Página Principal)
- `/emotion/`: **Análisis de Emociones**
- `/api/predict_tumor`: Endpoint API para predicción de tumores.
- `/api/predict_emotion`: Endpoint API para predicción de emociones (Mock).
