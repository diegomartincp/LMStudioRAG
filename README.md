Instalar FAISS
pip install faiss-cpu --only-binary=:all:

# Sistema RAG (Retrieval-Augmented Generation) con LM Studio y Flask

Este proyecto implementa un sistema **RAG (Retrieval-Augmented Generation)** que permite procesar documentos almacenados localmente (incluyendo subcarpetas) y realizar consultas sobre ellos utilizando un modelo LLM cargado en **LM Studio**.

## Características

- Procesa documentos PDF almacenados en una carpeta y sus subcarpetas.
- Utiliza **FAISS** como base de datos vectorial para realizar búsquedas por similitud.
- Integra un modelo LLM local cargado en **LM Studio** para generar respuestas basadas en los documentos.
- Proporciona una API REST con Flask para interactuar con el sistema.

---

## Requisitos

1. Python 3.8 o superior.
2. LM Studio instalado y configurado con un modelo compatible (por ejemplo, Llama 2 o Vicuna).
3. Dependencias especificadas en `requirements.txt`.

---

## Instalación

1. Clona este repositorio:
   git clone <URL-del-repositorio>
   cd <nombre-del-repositorio>

2. Crea un entorno virtual e instálalo:
   python -m venv venv
   source venv/bin/activate # En Windows: venv\Scripts\activate
   pip install -r requirements.txt

3. Configura LM Studio:

- Descarga e instala LM Studio desde su [página oficial](https://lmstudio.ai).
- Carga un modelo compatible y habilita la API local.

---

## Uso

1. Coloca tus documentos PDF en una carpeta principal (puede incluir subcarpetas). Configura la ruta en el archivo principal del proyecto:
   DOCUMENTS_FOLDER = r"ruta/a/tu/carpeta"

2. Ejecuta el servidor Flask:
   python rag.py

3. Envía consultas al sistema mediante una herramienta como `curl` o Postman:
   curl -X POST http://localhost:8080/query
   -H "Content-Type: application/json"
   -d '{"query": "¿Qué dice el documento sobre Kubernetes?"}'

---

## Estructura del Proyecto

├── rag.py # Código principal del sistema RAG.
├── requirements.txt # Dependencias del proyecto.
├── README.md # Documentación del proyecto.
└── <documentos> # Carpeta con los documentos PDF a procesar.

---

## Notas

- Si tienes problemas con la ejecución, verifica que LM Studio esté corriendo y que la API local esté habilitada.
- Puedes ajustar parámetros como el tamaño de los fragmentos (`chunk_size`) o el número de resultados relevantes (`k`) dentro del código.

---

## Contribuciones

Las contribuciones son bienvenidas. Si deseas colaborar, abre un issue o envía un pull request.

---

## Licencia

Este proyecto está bajo la licencia MIT.
