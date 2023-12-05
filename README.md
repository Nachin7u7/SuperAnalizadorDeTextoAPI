# SuperAnalizadorDeTextoAPI
## Introducción

Este servicio FastAPI proporciona capacidades de procesamiento del lenguaje natural (NLP), incluyendo análisis de sentimiento y análisis de texto utilizando modelos preentrenados. Además, incluye un análisis mejorado utilizando el modelo GPT-3.5-turbo de OpenAI.

## Instalación

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/Nachin7u7/SuperAnalizadorDeTextoAPI
   cd SuperAnalizadorDeTextoAPI
   ```
2. Instalar dependencias
   ```bash
   pip install -r requirements.txt 
   ```
## Usos
1. Verificar Estado del Servicio
   ```bash
   curl -X GET "http://127.0.0.1:8000/status"
   ```
3. Análisis de Sentimiento
   ```bash
   curl -X POST "http://127.0.0.1:8000/sentiment" -H "Content-Type: application/json" -d '{"text": "Tu texto para análisis de sentimiento"}'
   ```
5. Análisis de Texto
   ```bash
   curl -X POST "http://127.0.0.1:8000/analysis" -H "Content-Type: application/json" -d '{"text": "Tu texto para análisis"}'
   ```
7. Análisis Mejorado con GPT-3.5-turbo
   ```bash
   curl -X POST "http://127.0.0.1:8000/analysis_v2" -H "Content-Type: application/json" -d '{"text": "Tu texto para análisis", "prompt": "Tu prompt para GPT-3.5-turbo"}'
   ```
9. Generar Informes
   ```bash
   curl -X GET "http://127.0.0.1:8000/reports"
   ```
## Docker
```bash
docker build -t fastapi-nlp-service .
docker run -p 8000:8000 fastapi-nlp-service
```


## Autores
  Ricardo I. Valencia (Nachin7u7)  
  Alejandra Garcia (aleandy7)
