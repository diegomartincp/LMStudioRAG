import os
import numpy as np
import requests
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# Configuración inicial
# DOCUMENTS_FOLDER = "documents"
DOCUMENTS_FOLDER = "G:/Mi unidad/0. Master pentesting"
LM_STUDIO_API_URL = "http://localhost:8081/v1/completions"  # URL de la API local de LM Studio
MODEL_NAME = "deepseek-r1-distill-qwen-7b"  # Cambia esto al nombre del modelo cargado en LM Studio

# Inicializar el modelo de embeddings y la base de datos vectorial FAISS
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = None  # Se inicializará más adelante
chunks = []  # Lista para almacenar los fragmentos procesados


def process_documents(folder_path):
    global chunks, index

    # Lista para almacenar todos los documentos encontrados
    documents = []

    # Recorrer recursivamente la carpeta y sus subcarpetas
    for root, dirs, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".pdf"):  # Procesar solo archivos PDF
                file_path = os.path.join(root, file_name)
                loader = PyPDFLoader(file_path)
                print("Found document {file_name}")
                documents.extend(loader.load())

    # Dividir los documentos en fragmentos (chunks)
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    # Generar embeddings para los fragmentos
    embeddings = [embedding_model.encode(chunk.page_content) for chunk in chunks]

    # Crear e indexar los embeddings en FAISS
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    print(f"Indexado {len(embeddings)} fragmentos en FAISS.")


# Función para interactuar con LM Studio mediante su API local
def query_lm_studio(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 200,
    }
    response = requests.post(LM_STUDIO_API_URL, json=payload)
    response.raise_for_status()  # Lanza un error si hay un problema HTTP
    result = response.json()
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["text"]  # Accede al texto generado
        
    # Si no hay texto en las opciones, lanza un error
    raise ValueError("La respuesta no contiene texto válido.")


# Pipeline RAG: Recuperar contexto relevante y generar respuesta con LLM
def rag_pipeline(user_query):
    # Generar embedding de la consulta del usuario
    query_embedding = embedding_model.encode(user_query)

    # Buscar los fragmentos más similares en FAISS
    distances, indices = index.search(np.array([query_embedding]), k=5)
    relevant_chunks = [chunks[i] for i in indices[0]]

    # Crear el contexto a partir de los fragmentos relevantes
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])

    # Crear el prompt final para el modelo LLM
    prompt = f"Contexto:\n{context}\n\nPregunta: {user_query}\nRespuesta:"

    # Consultar al modelo en LM Studio y devolver la respuesta generada
    return query_lm_studio(prompt)


# Configurar una API Flask para interactuar con el sistema RAG
app = Flask(__name__)


@app.route("/query", methods=["POST"])
def handle_query():
    try:
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Solicitud inválida"}), 400

        user_query = data["query"]
        response = rag_pipeline(user_query)
        return jsonify({"response": response})
    except Exception as e:
        # Registrar el error en la consola
        app.logger.error(f"Error al procesar la consulta: {e}")
        # Devolver el traceback como respuesta HTTP
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.errorhandler(500)
def internal_server_error(e):
    app.logger.error(f"Error interno: {e}")
    return jsonify({"error": "Error interno del servidor"}), 500


if __name__ == "__main__":
    print("Procesando documentos...")
    process_documents(DOCUMENTS_FOLDER)  # Procesar e indexar los documentos al iniciar el servidor

    print("Iniciando servidor Flask...")
    app.debug = True  # Activa el modo de depuración
    app.run(port=8080)
