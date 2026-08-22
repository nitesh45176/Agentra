import streamlit as st
import os
import uuid

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from langgraph.types import Command

from backend import (
    chatbot,
    get_all_threads,
    insert_rag_document,
    delete_thread,
)


# ============================================================
# Helper Functions
# ============================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():

    st.session_state["thread_id"] = generate_thread_id()

    st.session_state["messages"] = []

    # Reset document information
    st.session_state["uploaded_document"] = None

    # Reset HITL state
    st.session_state["pending_approval"] = None

    add_thread(st.session_state["thread_id"])


def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


def _shorten_title(text, max_len=40):
    """Collapse whitespace/newlines and cap length for a tidy sidebar label."""

    text = " ".join(text.strip().split())

    if len(text) > max_len:
        text = text[:max_len].rstrip() + "…"

    return text


def get_thread_title(thread_id):
    """
    Return a short, human-friendly title for a thread (its first user
    message), instead of showing the raw UUID in the sidebar.
    """

    # Active thread: derive live from in-memory messages — cheap, and
    # always reflects the latest state (e.g. right after the very
    # first message is sent, before any rerun).
    if thread_id == st.session_state.get("thread_id"):

        for message in st.session_state.get("messages", []):

            if message["role"] == "user" and message["content"]:
                return _shorten_title(message["content"])

        return "New chat"

    # Other threads: cache the title so we don't reload the full
    # conversation from the checkpointer on every single rerun.
    titles = st.session_state["thread_titles"]

    if thread_id in titles:
        return titles[thread_id]

    title = "New chat"

    for message in load_conversation(thread_id):

        if isinstance(message, HumanMessage) and message.content:

            content = message.content

            if not isinstance(content, str):

                if isinstance(content, list):

                    content = "".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict)
                    )

                else:
                    content = str(content)

            if content.strip():
                title = _shorten_title(content)

            break

    titles[thread_id] = title

    return title


# ============================================================
# Document Upload
# ============================================================

def process_uploaded_file(uploaded_file):
    """
    Save an uploaded PDF to disk and index it for RAG.
    """

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        uploaded_file.name
    )

    try:

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner(
            f"Processing {uploaded_file.name}..."
        ):

            num_chunks = insert_rag_document(
                file_path
            )

        st.session_state["uploaded_document"] = (
            uploaded_file.name
        )

        st.toast(
            f"📄 {uploaded_file.name} processed "
            f"({num_chunks} chunks)",
            icon="✅",
        )

        return True

    except Exception as e:

        st.toast(
            f"Failed to process {uploaded_file.name}: {str(e)}",
            icon="❌",
        )

        return False


# ============================================================
# Resume HITL
# ============================================================

def resume_with_approval(decision):
    """
    Resume the paused LangGraph execution.

    decision should be:
        "yes"
        "no"
    """

    config = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        },
        "metadata": {
            "thread_id": st.session_state["thread_id"]
        },
        "run_name": "chat_trace",
    }

    # Clear the current approval request before resuming. It will only
    # be set again below if the resumed run hits a brand-new interrupt.
    # (Without this, an old approval value that's never explicitly
    # cleared mid-stream would stick around even after the request is
    # fully resolved, leaving the approval box stuck on screen.)
    st.session_state["pending_approval"] = None

    used_tools = []

    def stream_resumed_response():

        for mode, data in chatbot.stream(

            Command(resume=decision),

            config=config,

            stream_mode=[
                "messages",
                "updates",
            ],
        ):

            # =========================================
            # GRAPH UPDATES
            # =========================================

            if mode == "updates":

                if "__interrupt__" in data:

                    interrupts = data["__interrupt__"]

                    if interrupts:

                        interrupt_value = interrupts[0].value

                        st.session_state["pending_approval"] = interrupt_value

                    continue

            # =========================================
            # MESSAGE STREAM
            # =========================================

            if mode != "messages":
                continue

            message_chunk, metadata = data

            # -----------------------------------------
            # Tool Message
            # -----------------------------------------

            if isinstance(message_chunk, ToolMessage):

                tool_name = getattr(message_chunk, "name", None)

                if tool_name and tool_name not in used_tools:
                    used_tools.append(tool_name)

                continue

            # -----------------------------------------
            # AI Message
            # -----------------------------------------

            if (
                isinstance(message_chunk, AIMessage)
                and message_chunk.content
                and not getattr(message_chunk, "tool_calls", None)
            ):

                if isinstance(message_chunk.content, str):
                    text = message_chunk.content

                elif isinstance(message_chunk.content, list):
                    text = "".join(
                        block.get("text", "")
                        for block in message_chunk.content
                        if isinstance(block, dict)
                    )

                else:
                    text = str(message_chunk.content)

                if text:
                    yield text

    # ================================================
    # Display Assistant Response
    # ================================================

    with st.chat_message("assistant"):

        ai_message = st.write_stream(stream_resumed_response())

        if used_tools:
            st.caption("🔧 Tool(s) used: " + ", ".join(used_tools))

    # ================================================
    # Save Assistant Response
    # ================================================

    if ai_message:

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": ai_message,
            }
        )

    # HITL request has been handled. If the resumed run hit a brand-new
    # interrupt, pending_approval will already have been set again
    # above during the stream — leave it as-is in that case.
    if not st.session_state.get("pending_approval") and not ai_message:
        st.toast(
            "The assistant didn't return a response after your "
            "decision. Try sending a new message.",
            icon="⚠️",
        )


# ============================================================
# Session State Initialization
# ============================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_threads()

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "uploaded_document" not in st.session_state:
    st.session_state["uploaded_document"] = None

if "pending_approval" not in st.session_state:
    st.session_state["pending_approval"] = None

if "thread_titles" not in st.session_state:
    st.session_state["thread_titles"] = {}

# Add current thread
add_thread(st.session_state["thread_id"])


# ============================================================
# Main Title
# ============================================================

st.title("Agentra")

# Sidebar styling: a real logo heading (not a button) for the brand
# name, and a lighter "icon button" look for the small "+" new-chat
# icon and the per-chat delete buttons — transparent by default, with
# a colored hover. Streamlit exposes a "st-key-<key>" class on an
# element's wrapper when it has a `key=`, so each rule below only
# touches the button it's meant to.
st.markdown(
    """
    <style>
    .agentra-logo {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: 0.3px;
        margin: 0;
        line-height: 1.2;
        background: linear-gradient(90deg, #7C6CF2, #35C9E1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .agentra-tagline {
        color: #9aa0a6;
        font-size: 0.78rem;
        margin: 2px 0 0 0;
    }
    [class*="st-key-new_chat_icon"] button {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: #9aa0a6 !important;
        box-shadow: none !important;
        font-size: 1.1rem !important;
    }
    [class*="st-key-new_chat_icon"] button:hover {
        color: #35C9E1 !important;
        background: rgba(53, 201, 225, 0.12) !important;
        border-color: rgba(53, 201, 225, 0.4) !important;
    }
    [class*="st-key-delete_"] button {
        background: transparent !important;
        border: 1px solid transparent !important;
        color: #9aa0a6 !important;
        box-shadow: none !important;
    }
    [class*="st-key-delete_"] button:hover {
        color: #ff4b4b !important;
        background: rgba(255, 75, 75, 0.12) !important;
        border-color: rgba(255, 75, 75, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar - Branding / New Chat
# ============================================================

logo_col, new_chat_col = st.sidebar.columns([5, 1])

with logo_col:

    st.markdown(
        '<p class="agentra-logo">🧭 Agentra</p>'
        '<p class="agentra-tagline">Your AI research &amp; trading assistant</p>',
        unsafe_allow_html=True,
    )

with new_chat_col:

    if st.button("➕", key="new_chat_icon", help="Start a new chat"):

        reset_chat()

        st.rerun()


# ============================================================
# Sidebar - Document (only shown once a document is active)
# ============================================================

if st.session_state["uploaded_document"]:

    st.sidebar.info(
        "📄 Active document:\n\n" + st.session_state["uploaded_document"]
    )

    if st.sidebar.button("Clear document"):
        st.session_state["uploaded_document"] = None
        st.rerun()


# ============================================================
# Sidebar - Chat History
# ============================================================

st.sidebar.title("Chats")

for thread_id in st.session_state["chat_threads"][::-1]:

    is_active = thread_id == st.session_state["thread_id"]

    title = get_thread_title(thread_id)

    col1, col2 = st.sidebar.columns([5, 1])

    # --------------------------------------------------------
    # Open Chat
    # --------------------------------------------------------

    with col1:

        if st.button(
            ("💬 " if is_active else "") + title,
            key=f"chat_{thread_id}",
            help=thread_id,
            use_container_width=True,
            type="primary" if is_active else "secondary",
        ):

            st.session_state["thread_id"] = thread_id

            messages = load_conversation(thread_id)

            temp_messages = []

            for message in messages:

                if isinstance(message, HumanMessage):
                    role = "user"

                elif isinstance(message, AIMessage):
                    role = "assistant"

                else:
                    continue

                content = message.content

                if not isinstance(content, str):

                    if isinstance(content, list):

                        content = "".join(
                            block.get("text", "")
                            for block in content
                            if isinstance(block, dict)
                        )

                    else:
                        content = str(content)

                temp_messages.append(
                    {
                        "role": role,
                        "content": content,
                    }
                )

            st.session_state["messages"] = temp_messages

            # Clear any old HITL request
            st.session_state["pending_approval"] = None

            st.rerun()

    # --------------------------------------------------------
    # Delete Chat
    # --------------------------------------------------------

    with col2:

        if st.button(
            "🗑️",
            key=f"delete_{thread_id}",
            help="Delete this chat",
            use_container_width=True,
        ):

            delete_thread(thread_id)

            if thread_id in st.session_state["chat_threads"]:
                st.session_state["chat_threads"].remove(thread_id)

            # Drop its cached title too
            st.session_state["thread_titles"].pop(thread_id, None)

            # If current chat was deleted
            if st.session_state["thread_id"] == thread_id:

                new_thread_id = generate_thread_id()

                st.session_state["thread_id"] = new_thread_id
                st.session_state["messages"] = []
                st.session_state["pending_approval"] = None

                add_thread(new_thread_id)

            st.rerun()


# ============================================================
# Display Current Chat
# ============================================================

for message in st.session_state["messages"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ============================================================
# HITL APPROVAL UI
# ============================================================

if st.session_state["pending_approval"]:

    approval = st.session_state["pending_approval"]

    st.warning("⚠️ Human approval required")

    # --------------------------------------------------------
    # Dictionary-based interrupt
    # --------------------------------------------------------

    if isinstance(approval, dict):

        question = approval.get(
            "question", "The agent is requesting approval."
        )

        instruction = approval.get("instruction", "Do you approve?")

        st.write(f"**Question:** {question}")
        st.write(f"**{instruction}**")

    # --------------------------------------------------------
    # String-based interrupt
    # --------------------------------------------------------

    else:

        st.write(str(approval))

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "✅ Approve",
            key="hitl_approve",
            use_container_width=True,
        ):

            resume_with_approval("yes")
            st.rerun()

    with col2:

        if st.button(
            "❌ Reject",
            key="hitl_reject",
            use_container_width=True,
        ):

            resume_with_approval("no")
            st.rerun()


# ============================================================
# Chat Input
# ============================================================

chat_value = st.chat_input(
    "Type here, or attach a PDF with the 📎 icon",
    accept_file="multiple",
    file_type=["pdf"],
)


# ============================================================
# Process User Input
# ============================================================

if chat_value:

    user_input = chat_value.text
    attached_files = chat_value.files or []

    # --------------------------------------------------------
    # Process PDFs
    # --------------------------------------------------------

    for f in attached_files:
        process_uploaded_file(f)

    # --------------------------------------------------------
    # File uploaded without text
    # --------------------------------------------------------

    if not user_input and attached_files:

        file_names = ", ".join(f.name for f in attached_files)
        user_input = f"[Uploaded document(s): {file_names}]"

    # --------------------------------------------------------
    # User entered text
    # --------------------------------------------------------

    if user_input:

        # ----------------------------------------------------
        # Display User Message
        # ----------------------------------------------------

        st.session_state["messages"].append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.chat_message("user"):
            st.write(user_input)

        # ----------------------------------------------------
        # LangGraph Configuration
        # ----------------------------------------------------

        config = {
            "configurable": {
                "thread_id": st.session_state["thread_id"]
            },
            "metadata": {
                "thread_id": st.session_state["thread_id"]
            },
            "run_name": "chat_trace",
        }

        # ----------------------------------------------------
        # Track Tools
        # ----------------------------------------------------

        used_tools = []

        # ----------------------------------------------------
        # Stream Response
        # ----------------------------------------------------

        def stream_response():

            for mode, data in chatbot.stream(

                {"messages": [HumanMessage(content=user_input)]},

                config=config,

                stream_mode=[
                    "messages",
                    "updates",
                ],
            ):

                # =============================================
                # GRAPH UPDATES
                # =============================================

                if mode == "updates":

                    # ---------------------------------------
                    # HITL interrupt
                    # ---------------------------------------

                    if "__interrupt__" in data:

                        interrupts = data["__interrupt__"]

                        if interrupts:

                            interrupt_value = interrupts[0].value

                            st.session_state["pending_approval"] = interrupt_value

                        continue

                # =============================================
                # MESSAGE STREAM
                # =============================================

                if mode != "messages":
                    continue

                message_chunk, metadata = data

                # ---------------------------------------------
                # Tool Message
                # ---------------------------------------------

                if isinstance(message_chunk, ToolMessage):

                    tool_name = getattr(message_chunk, "name", None)

                    if tool_name and tool_name not in used_tools:
                        used_tools.append(tool_name)

                    continue

                # ---------------------------------------------
                # AI Message
                # ---------------------------------------------

                if (
                    isinstance(message_chunk, AIMessage)
                    and message_chunk.content
                    and not getattr(message_chunk, "tool_calls", None)
                ):

                    # Normal string
                    if isinstance(message_chunk.content, str):
                        text = message_chunk.content

                    # Structured content
                    elif isinstance(message_chunk.content, list):
                        text = "".join(
                            block.get("text", "")
                            for block in message_chunk.content
                            if isinstance(block, dict)
                        )

                    else:
                        text = str(message_chunk.content)

                    if text:
                        yield text

        # ------------------------------------------------------
        # Display Assistant
        # ------------------------------------------------------

        with st.chat_message("assistant"):

            ai_message = st.write_stream(stream_response())

            if used_tools:
                st.caption("🔧 Tool(s) used: " + ", ".join(used_tools))

        # ------------------------------------------------------
        # Save Assistant Message
        # ------------------------------------------------------

        if ai_message:

            st.session_state["messages"].append(
                {
                    "role": "assistant",
                    "content": ai_message,
                }
            )

        # ------------------------------------------------------
        # If the run paused for human approval, rerun immediately
        # so the HITL approval UI shows up right away instead of
        # waiting for the user's next message.
        # ------------------------------------------------------

        if st.session_state.get("pending_approval"):
            st.rerun()

        # ------------------------------------------------------
        # Safety net: if the run produced no visible text and did
        # not pause for approval either, something went wrong
        # silently (e.g. the model looped back into a tool call
        # with no final answer). Surface it instead of looking
        # like a dead button.
        # ------------------------------------------------------

        elif not ai_message:
            st.toast(
                "The assistant didn't return a response. "
                "Try rephrasing or sending the message again.",
                icon="⚠️",
            )