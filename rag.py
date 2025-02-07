import os
import numpy as np
import requests
from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import torch
from concurrent.futures import ThreadPoolExecutor
import pickle

# Configuración inicial
DOCUMENTS_FOLDER = r"G:/Mi unidad/0. Master pentesting"
LM_STUDIO_API_URL = "http://localhost:8081/v1/completions"  # URL de la API local de LM Studio
MODEL_NAME = "deepseek-r1-distill-qwen-7b"  # Cambia esto al nombre del modelo cargado en LM Studio
INDEX_FILE = "faiss_index.bin"  # Nombre del archivo donde se almacenará el índice
CHUNKS_FILE = "chunks.pkl"  # Archivo donde se guardarán los fragmentos

# Inicializar el modelo de embeddings con soporte para GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
print(f"Usando dispositivo: {device}")

index = None  # Se inicializará más adelante
chunks = []  # Lista para almacenar los fragmentos procesados


def generate_embeddings_parallel(chunks):
    """
    Genera embeddings en paralelo utilizando múltiples hilos.
    """
    def embed_chunk(chunk):
        return embedding_model.encode(chunk.page_content)

    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(embed_chunk, chunks))
    
    return embeddings


def process_documents_with_progress(folder_path):
    global index, chunks

    all_files = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith(".pdf"):  # Filtrar solo archivos PDF
                file_path = os.path.join(root, file_name)
                all_files.append(file_path)

    total_files = len(all_files)
    if total_files == 0:
        print("No se encontraron archivos PDF.")
        return

    print(f"Total de documentos a procesar: {total_files}")

    for i, file_path in enumerate(all_files, start=1):
        file_name = os.path.basename(file_path)
        print(f"Procesando archivo: {file_name} ({i}/{total_files})")

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks.extend(splitter.split_documents(documents))

    print(f"Total de fragmentos generados: {len(chunks)}")

    print("Generando embeddings...")
    embeddings = [embedding_model.encode(chunk.page_content) for chunk in chunks]

    print("Creando índice FAISS...")
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Guardar el índice en disco
    print(f"Guardando el índice en {INDEX_FILE}...")
    faiss.write_index(index, INDEX_FILE)
    
    # Guardar los fragmentos en disco
    print(f"Guardando los fragmentos en {CHUNKS_FILE}...")
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

    print("Índice y fragmentos guardados correctamente.")


def load_or_create_index(folder_path):
    global index, chunks

    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print(f"Cargando índice desde {INDEX_FILE}...")
        index = faiss.read_index(INDEX_FILE)
        print("Índice cargado correctamente.")

        print(f"Cargando fragmentos desde {CHUNKS_FILE}...")
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
        print(f"Se cargaron {len(chunks)} fragmentos.")
    
    else:
        print("No se encontró un índice existente. Procesando documentos...")
        process_documents_with_progress(folder_path)


def query_lm_studio(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 200,
    }
    print(f"Prompt enviado a LM Studio:\n{prompt}")  # Agrega esta línea para depurar
    response = requests.post(LM_STUDIO_API_URL, json=payload)
    response.raise_for_status()

    result = response.json()
    
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["text"]

    raise ValueError("La respuesta no contiene texto válido.")


def rag_pipeline(user_query):
    global index, chunks

    # Verificar si el índice está inicializado y contiene datos
    if index is None or index.ntotal == 0:
        raise ValueError("El índice FAISS está vacío o no ha sido inicializado. Asegúrate de procesar los documentos primero.")

    if not chunks:
        raise ValueError("Los fragmentos no están cargados. Asegúrate de cargar o procesar los documentos primero.")

    # Generar embedding de la consulta del usuario
    query_embedding = embedding_model.encode(user_query)

    # Buscar los fragmentos más similares en FAISS
    distances, indices = index.search(np.array([query_embedding]), k=5)

    # Verificar si se encontraron resultados
    if len(indices[0]) == 0 or all(i == -1 for i in indices[0]):
        return "No se encontraron fragmentos relevantes para responder a tu consulta."

    relevant_chunks = [chunks[i] for i in indices[0]]

    # Crear el contexto a partir de los fragmentos relevantes (limitado a 1000 caracteres)
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])[:1000]

    if not context.strip():  # Si el contexto está vacío o solo tiene espacios
        return "No se pudo generar un contexto relevante para la consulta."

    # Crear el prompt final para el modelo LLM
    prompt = f"Contexto:\n{context}\n\nPregunta: {user_query}\nRespuesta:"

    return query_lm_studio(prompt)


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
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


if __name__ == "__main__":
    print("Cargando o creando índice...")
    
    load_or_create_index(DOCUMENTS_FOLDER)  # Cargar o crear el índice
    
    print("Iniciando servidor Flask...")
    
    app.run(port=8080)
