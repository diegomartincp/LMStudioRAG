import os
import json
import yaml
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify


# Configuración inicial
DOCUMENTS_FOLDER = r"./documents"
OLLAMA_BASE_URL="http://10.95.118.77:11434"
MODEL_API_URL = "http://10.95.118.77:11434/v1/chat/completions"
MODEL_NAME = "llama3.1:8b"
CHROMA_DIR = "chroma_db"
EMBEDDING_MODEL="nomic-embed-text:latest"
MAX_CONTEXT_LENGTH= 5000
MAX_RESPONSE_TOKENS=1000

# Inicializar el modelo de embedding
embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL 
)


vectorstore = None  # Se inicializará más adelante

# Función que carga los documentos en función de su tipo
def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()
    elif ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Mejor: usar RecursiveJsonSplitter para fragmentar el JSON semánticamente
            splitter = RecursiveJsonSplitter(max_chunk_size=1000)
            json_chunks = splitter.split_json(data, True)
            documents = [
                Document(page_content=json.dumps(chunk, indent=2), metadata={"source": file_path})
                for chunk in json_chunks
            ]
        return documents
    elif ext in [".yaml", ".yml"]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return [Document(page_content=yaml.dump(data, allow_unicode=True), metadata={"source": file_path})]
    else:
        raise ValueError(f"Tipo de archivo no soportado: {ext}")

# Función que procesa los documentos y los almacena en la base de datos
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
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )
    vectorstore.persist()
    print("Índice y fragmentos guardados correctamente.")

#Función para cargar o crear el índice de la base de datos vectorial
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

# Función que mnda la query al modelo LLM
def query_lm_studio(prompt):
    import requests
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": MAX_RESPONSE_TOKENS,
    }
    print(f"Prompt enviado al modelo LLM:\n{prompt}")
    response = requests.post(MODEL_API_URL, json=payload)
    print(f"Response:\n{response}")
    response.raise_for_status()
    result = response.json()
    if "choices" in result and len(result["choices"]) > 0:
        return result["choices"][0]["message"]["content"]
    raise ValueError("La respuesta no contiene texto válido.")

# Función principal que ejecuta el pipeline de RAG y busca coincidenancias en la base de datos para crear el contexto
def rag_pipeline(user_query):
    global vectorstore
    if vectorstore is None:
        raise ValueError("La base de datos Chroma no ha sido inicializada. Asegúrate de procesar los documentos primero.")

    # Buscar los fragmentos más similares en Chroma
    relevant_docs = vectorstore.similarity_search(user_query, k=5)
    if not relevant_docs:
        return "No se encontraron fragmentos relevantes para responder a tu consulta."

    context = "\n".join([doc.page_content for doc in relevant_docs])[:MAX_CONTEXT_LENGTH]
    if not context.strip():
        return "No se pudo generar un contexto relevante para la consulta."

    prompt = f"""
                Por favor, responde a la siguiente pregunta basándote únicamente en la información proporcionada en el contexto.
                Si la respuesta no está en el contexto, responde: "No tengo suficiente información para responder a esta pregunta."

                Contexto:
                {context}

                Pregunta: {user_query}

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