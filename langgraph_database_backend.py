import os
from pathlib import Path
import sqlite3
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


# Default to a valid HF chat model; override in .env via HF_MODEL_ID.
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

endpoint = HuggingFaceEndpoint(
    repo_id=HF_MODEL_ID,
)
llm = ChatHuggingFace(llm=endpoint)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
DB_PATH = Path(os.getenv("CHATBOT_DB_PATH", str(BASE_DIR / "chatbot.db")))
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
    return next(
        checkpointer.list({"configurable": {"thread_id": str(thread_id)}}),
        None,
    )


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


try:
    retriever, DOC_COUNT, INIT_ERROR = _initialize_retriever()
except Exception as exc:
    retriever = None
    DOC_COUNT = 0
    INIT_ERROR = f"Failed to initialize FAISS retriever: {exc}"

RAG_ENABLED = retriever is not None


class ChatState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    rag_context: str
    rag_sources: list[str]
    chat_title: str


def retrieve_node(state: ChatState):
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


def chat_node(state: ChatState):
    messages = state["messages"]
    rag_context = state.get("rag_context", "")
    chat_title = state.get("chat_title", "")

    if not chat_title:
        first_user_message = next(
            (message.content for message in messages if isinstance(
                message, HumanMessage)),
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


graph = StateGraph(ChatState)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "retrieve_node")
graph.add_edge("retrieve_node", "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)


def get_last_rag_sources(thread_id):
    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}})
    return state.values.get("rag_sources", [])


def get_rag_info():
    return {
        "enabled": RAG_ENABLED,
        "data_dir": str(DATA_DIR),
        "doc_count": DOC_COUNT,
        "error": INIT_ERROR,
    }


def get_thread_title(thread_id: str) -> str:
    checkpoint = _get_latest_checkpoint(thread_id)
    if checkpoint is not None:
        metadata_title = checkpoint.metadata.get("chat_title", "")
        if metadata_title:
            return metadata_title

    state = chatbot.get_state(
        config={"configurable": {"thread_id": thread_id}})
    title = state.values.get("chat_title", "")
    if title:
        return title

    messages = state.values.get("messages", [])
    first_user_message = next(
        (message.content for message in messages if isinstance(message, HumanMessage)),
        "",
    )
    if first_user_message:
        return build_chat_title(first_user_message)

    return "New Chat"


def clear_all_chat_data():
    cursor = conn.cursor()
    cursor.execute("DELETE FROM writes")
    cursor.execute("DELETE FROM checkpoints")
    conn.commit()
    return {"status": "cleared"}


def clear_chat_thread(thread_id: str):
    checkpointer.delete_thread(str(thread_id))
    return {"status": "cleared", "thread_id": str(thread_id)}
