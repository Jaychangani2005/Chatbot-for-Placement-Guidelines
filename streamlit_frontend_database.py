import streamlit as st
from langgraph_database_backend import (
    chatbot,
    clear_all_chat_data,
    build_chat_title,
    get_last_rag_sources,
    get_rag_info,
    get_thread_title,
    retrieve_all_threads,
)
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# **************************************** utility functions *************************

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id


def start_new_chat():
    new_thread_id = generate_thread_id()
    st.session_state['thread_id'] = new_thread_id
    add_thread(new_thread_id)
    st.session_state['message_history'] = []


def confirm_clear_all_data():
    clear_all_chat_data()
    st.session_state['message_history'] = []
    st.session_state['thread_id'] = generate_thread_id()
    st.session_state['chat_threads'] = []
    add_thread(st.session_state['thread_id'])
    st.rerun()


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', [])


# **************************************** Session Setup ******************************
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()

add_thread(st.session_state['thread_id'])


# **************************************** Sidebar UI *********************************

st.sidebar.title('Placement Policy Assistant')

rag_info = get_rag_info()
# if rag_info['enabled']:
#     st.sidebar.success(
#         f"FAISS knowledge base ready ({rag_info['doc_count']} document(s))")
# else:
#     st.sidebar.warning('Knowledge base not loaded')
#     st.sidebar.caption(f"Reason: {rag_info['error']}")
#     st.sidebar.caption(f"Data folder: {rag_info['data_dir']}")

if st.sidebar.button('New Chat'):
    start_new_chat()

@st.dialog('Clear all chat data?')
def clear_all_dialog():
    st.warning('This will permanently delete all stored conversations from the app.')
    st.write('This action cannot be undone.')
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button('Yes, clear everything', type='primary'):
            confirm_clear_all_data()
    with col_no:
        if st.button('Cancel'):
            st.rerun()


if st.sidebar.button('Clear All Chat Data'):
    clear_all_dialog()

st.sidebar.header('My Conversations')

for thread_id in st.session_state['chat_threads'][::-1]:
    thread_title = get_thread_title(str(thread_id))
    if st.sidebar.button(thread_title, key=f"thread-{thread_id}"):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = 'user'
            else:
                role = 'assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages


# **************************************** Main UI ************************************

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])
        if message.get('sources'):
            with st.expander('Sources used'):
                for source in message['sources']:
                    st.markdown(f"- {source}")

user_input = st.chat_input('Type here')

if user_input:

    # first add the message to message_history
    st.session_state['message_history'].append(
        {'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    # CONFIG = {'configurable': {'thread_id': st.session_state['thread_id']}}

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {
            "thread_id": st.session_state["thread_id"],
            "chat_title": (
                build_chat_title(user_input)
                if get_thread_title(st.session_state["thread_id"]) == "New Chat"
                else get_thread_title(st.session_state["thread_id"])
            ),
        },
        "run_name": "chat_turn",
    }

    # first add the message to message_history
    with st.chat_message("assistant"):
        def ai_only_stream():
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, AIMessage):
                    # yield only assistant tokens
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        rag_sources = get_last_rag_sources(st.session_state['thread_id'])
        if rag_sources:
            with st.expander('Sources used'):
                for source in rag_sources:
                    st.markdown(f"- {source}")

    st.session_state['message_history'].append(
        {'role': 'assistant', 'content': ai_message, 'sources': rag_sources}
    )
