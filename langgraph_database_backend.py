import os
import json
from pathlib import Path
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = Path(os.getenv("CHATBOT_DB_PATH", str(BASE_DIR / "chatbot.db")))
ENABLE_RAG = os.getenv("ENABLE_RAG", "false").lower() == "true"
SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json"}
# DEFAULT_SYSTEM_PROMPT = (
#     "You are CHARUSAT Placement Guidelines Assistant. Answer only using the provided "
#     "placement-guidelines context when available. If the context does not contain the answer, "
#     "say that clearly and avoid guessing. Keep the answer concise, practical, and policy-focused."
# )

DEFAULT_SYSTEM_PROMPT = (
    "You are the CHARUSAT Placement Guidelines Assistant. "
    "Context Usage: Use only the provided placement guidelines context to answer queries. "
    "Missing Information: If the answer is not available in the context, say that clearly and avoid making assumptions or guesses. "
    "Response Style: say that clearly and avoid guessing, Keep answers concise, practical, and policy-focused."
)


def build_chat_title(text: str) -> str:
    cleaned_text = " ".join(text.split()).strip()

    if not cleaned_text:
        return "New Chat"

    words = cleaned_text.split()
    return " ".join(words[:4])


def _get_latest_checkpoint(thread_id: str):
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT metadata
                FROM checkpoints
                WHERE thread_id = ?
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (str(thread_id),),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {"metadata": row[0]}
    except sqlite3.Error:
        return None


def _read_checkpoint_metadata(thread_id: str) -> dict:
    checkpoint = _get_latest_checkpoint(thread_id)
    if checkpoint is None:
        return {}

    metadata = checkpoint.get("metadata", "")
    if isinstance(metadata, dict):
        return metadata

    if not metadata:
        return {}

    try:
        parsed = json.loads(metadata)
    except (TypeError, json.JSONDecodeError):
        return {}

    return parsed if isinstance(parsed, dict) else {}


def _read_thread_title(thread_id: str) -> str:
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT value
                FROM writes
                WHERE thread_id = ? AND channel = 'chat_title'
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (str(thread_id),),
            )
            row = cursor.fetchone()
            if row is None or row[0] is None:
                return ""

            value = row[0]
            try:
                import msgpack

                title = msgpack.unpackb(value, raw=False)
            except Exception:
                if isinstance(value, (bytes, bytearray)):
                    title = value.decode("utf-8", errors="ignore")
                else:
                    title = str(value)

            return str(title).strip()
    except sqlite3.Error:
        return ""


def _load_documents_from_dir(doc_dir: Path) -> list[Document]:
    if not doc_dir.exists():
        return []

    documents: list[Document] = []
    for file_path in doc_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={"source": str(file_path.relative_to(BASE_DIR))},
            )
        )
    return documents


def _build_retriever_from_documents(documents: list[Document]):
    if not documents:
        return None

    # Import here to avoid loading sentence-transformers/torch on cold start
    # when RAG is disabled in production environments.
    from langchain_huggingface import HuggingFaceEmbeddings

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def _initialize_retriever():
    documents = _load_documents_from_dir(DATA_DIR)
    retriever = _build_retriever_from_documents(documents)
    if retriever is None:
        return None, 0, f"No supported documents found in {DATA_DIR}"
    return retriever, len(documents), ""


retriever = None
DOC_COUNT = 0
INIT_ERROR = ""
RAG_ENABLED = False
_chatbot = None
_checkpointer = None


def _get_db_connection():
    global _checkpointer

    if _checkpointer is None:
        return sqlite3.connect(database=str(DB_PATH), check_same_thread=False)

    return _checkpointer.conn


def _build_chatbot():
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    hf_model_id = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")
    endpoint = HuggingFaceEndpoint(repo_id=hf_model_id)
    llm = ChatHuggingFace(llm=endpoint)

    class _ChatState(TypedDict, total=False):
        messages: Annotated[list[BaseMessage], add_messages]
        rag_context: str
        rag_sources: list[str]
        chat_title: str

    def retrieve_node(state: _ChatState):
        _ensure_retriever_initialized()

        if not RAG_ENABLED or retriever is None:
            return {"rag_context": "", "rag_sources": []}

        last_user_query = ""
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                last_user_query = message.content
                break

        if not last_user_query:
            return {"rag_context": "", "rag_sources": []}

        docs = retriever.invoke(last_user_query)
        context_chunks = [doc.page_content for doc in docs]
        sources = sorted({doc.metadata.get("source", "unknown") for doc in docs})

        return {
            "rag_context": "\n\n".join(context_chunks),
            "rag_sources": sources,
        }

    def chat_node(state: _ChatState):
        messages = state["messages"]
        rag_context = state.get("rag_context", "")
        chat_title = state.get("chat_title", "")

        if not chat_title:
            first_user_message = next(
                (message.content for message in messages if isinstance(message, HumanMessage)),
                "",
            )
            if first_user_message:
                chat_title = build_chat_title(first_user_message)
            else:
                chat_title = "New Chat"

        if rag_context:
            system_prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\nRetrieved context:\n{rag_context}"
        else:
            system_prompt = (
                f"{DEFAULT_SYSTEM_PROMPT}\n\nNo placement-guidelines context was retrieved for this turn. "
                "Answer only from the conversation so far."
            )

        try:
            response = llm.invoke([SystemMessage(content=system_prompt), *messages])
        except Exception as exc:
            error_text = str(exc)
            if "model_not_found" in error_text or "does not exist" in error_text:
                response = AIMessage(
                    content=(
                        "I could not reach the configured Hugging Face model. "
                        "Set HF_MODEL_ID to a valid model (for example, "
                        "meta-llama/Llama-3.1-8B-Instruct) and ensure your HF token has access."
                    )
                )
            else:
                response = AIMessage(
                    content=(
                        "The language model request failed. "
                        "Please verify HF_MODEL_ID and HF_TOKEN, then retry."
                    )
                )
        return {"messages": [response], "chat_title": chat_title}

    conn = sqlite3.connect(database=str(DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn=conn)

    graph = StateGraph(_ChatState)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "retrieve_node")
    graph.add_edge("retrieve_node", "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpointer)


def _get_chatbot():
    global _chatbot

    if _chatbot is None:
        _chatbot = _build_chatbot()

    return _chatbot


class _LazyChatbotProxy:
    def __getattr__(self, name):
        return getattr(_get_chatbot(), name)


chatbot = _LazyChatbotProxy()


def _ensure_retriever_initialized():
    global retriever, DOC_COUNT, INIT_ERROR, RAG_ENABLED

    if not ENABLE_RAG:
        RAG_ENABLED = False
        if not INIT_ERROR:
            INIT_ERROR = "RAG disabled by ENABLE_RAG=false"
        return

    if retriever is not None or RAG_ENABLED:
        return

    try:
        retriever, DOC_COUNT, INIT_ERROR = _initialize_retriever()
        RAG_ENABLED = retriever is not None
    except Exception as exc:
        retriever = None
        DOC_COUNT = 0
        INIT_ERROR = f"Failed to initialize FAISS retriever: {exc}"
        RAG_ENABLED = False


def retrieve_all_threads():
    try:
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id")
            return [row[0] for row in cursor.fetchall() if row and row[0]]
    except sqlite3.Error:
        return []


def get_last_rag_sources(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}})
    return state.values.get("rag_sources", [])


def get_rag_info():
    _ensure_retriever_initialized()

    return {
        "enabled": RAG_ENABLED,
        "data_dir": str(DATA_DIR),
        "doc_count": DOC_COUNT,
        "error": INIT_ERROR,
    }


def get_thread_title(thread_id: str) -> str:
    title = _read_thread_title(thread_id)
    if title:
        return title

    return "New Chat"


def clear_all_chat_data():
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM writes")
        cursor.execute("DELETE FROM checkpoints")
        conn.commit()
    return {"status": "cleared"}


def clear_chat_thread(thread_id: str):
    with sqlite3.connect(str(DB_PATH)) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (str(thread_id),))
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (str(thread_id),))
        conn.commit()
    return {"status": "cleared", "thread_id": str(thread_id)}
