import os
import json
import yaml

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
import torch

from flask import Flask, request, jsonify

# Configuración inicial
DOCUMENTS_FOLDER = r"./documents"
LM_STUDIO_API_URL = "http://10.95.118.77:11434/v1/chat/completions"
MODEL_NAME = "llama3.1:8b"
CHROMA_DIR = "chroma_db"  # Carpeta donde se almacenará la base de datos Chroma

class EmbeddingWrapper:
    def __init__(self, embed_func):
        self.embed_func = embed_func
    def embed_documents(self, texts):
        # Devuelve una lista de vectores para una lista de textos
        return [self.embed_func(text) for text in texts]
    def embed_query(self, text):
        # Devuelve el vector para una sola consulta
        return self.embed_func(text)

# Inicializar el modelo de embeddings con soporte para GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_model = SentenceTransformer(r"C:\Users\te03601\Documents\0. Code\all-MiniLM-L6-v2")
embedding_function = EmbeddingWrapper(lambda x: embedding_model.encode(x, show_progress_bar=False))
print(f"Usando dispositivo: {device}")

vectorstore = None  # Se inicializará más adelante

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Document(page_content=json.dumps(data, indent=2), metadata={"source": file_path})]
    elif ext in [".yaml", ".yml"]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [Document(page_content=yaml.dump(data, allow_unicode=True), metadata={"source": file_path})]
    else:
        raise ValueError(f"Tipo de archivo no soportado: {ext}")

def process_documents_with_progress(folder_path):
    global vectorstore
    all_files = []
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.endswith((".pdf", ".json", ".yaml", ".yml")):
                file_path = os.path.join(root, file_name)
                all_files.append(file_path)

    total_files = len(all_files)
    if total_files == 0:
        print("No se encontraron archivos PDF, JSON ni YAML.")
        return

    print(f"Total de documentos a procesar: {total_files}")

    chunks = []
    for i, file_path in enumerate(all_files, start=1):
        file_name = os.path.basename(file_path)
        print(f"Procesando archivo: {file_name} ({i}/{total_files})")
        try:
            documents = load_document(file_path)
            splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks.extend(splitter.split_documents(documents))
        except Exception as e:
            print(f"Error procesando {file_name}: {e}")

    print(f"Total de fragmentos generados: {len(chunks)}")

    # Crear la base de datos Chroma y guardar los fragmentos con embeddings
    print("Creando base de datos Chroma...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("Índice y fragmentos guardados correctamente.")

def load_or_create_index(folder_path):
    global vectorstore
    if os.path.exists(CHROMA_DIR) and os.path.exists(os.path.join(CHROMA_DIR, "chroma-collections.parquet")):
        print(f"Cargando índice desde {CHROMA_DIR}...")
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=lambda x: embedding_model.encode(x, show_progress_bar=False)
        )
        print("Índice cargado correctamente.")
    else:
        print("No se encontró un índice existente. Procesando documentos...")
        process_documents_with_progress(folder_path)

def query_lm_studio(prompt):
    import requests
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1000,
    }
    print(f"Prompt enviado a LM Studio:\n{prompt}")
    response = requests.post(LM_STUDIO_API_URL, json=payload)
    response.raise_for_status()
    result = response.json()
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    raise ValueError("La respuesta no contiene texto válido.")

def rag_pipeline(user_query):
    global vectorstore
    if vectorstore is None:
        raise ValueError("La base de datos Chroma no ha sido inicializada. Asegúrate de procesar los documentos primero.")

    # Buscar los fragmentos más similares en Chroma
    relevant_docs = vectorstore.similarity_search(user_query, k=5)
    if not relevant_docs:
        return "No se encontraron fragmentos relevantes para responder a tu consulta."

    context = "\n".join([doc.page_content for doc in relevant_docs])[:5000]
    if not context.strip():
        return "No se pudo generar un contexto relevante para la consulta."

    prompt = f"""
Por favor, responde a la siguiente pregunta basándote únicamente en la información proporcionada en el contexto.
Si la respuesta no está en el contexto, responde: "No tengo suficiente información para responder a esta pregunta."

Contexto:
{context}

Pregunta: {user_query}

Respuesta:
"""
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
    load_or_create_index(DOCUMENTS_FOLDER)
    print("Iniciando servidor Flask...")
    app.run(port=8080)