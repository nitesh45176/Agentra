import streamlit as st

# HumanMessage is used to convert the user's plain text
# into a LangChain message object.
from langchain_core.messages import HumanMessage

# Import the chatbot/LangGraph that we created
# in agentic_chatbot.py
from backend import chatbot


# =========================================================
# LANGGRAPH THREAD CONFIGURATION
# =========================================================

# thread_id identifies a particular conversation/thread.
#
# If we use a checkpointer in LangGraph, this ID allows
# LangGraph to remember the state of this conversation.
#
# For now, we are using a fixed ID "1".
thread_id = "1"


# LangGraph expects the thread ID inside the
# "configurable" dictionary.
#
# The structure is:
#
# {
#     "configurable": {
#         "thread_id": "1"
#     }
# }
config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# =========================================================
# STREAMLIT UI
# =========================================================

# Display the title at the top of the webpage.
st.title("Chatbot")


# =========================================================
# STREAMLIT SESSION STATE
# =========================================================

# Streamlit reruns the Python script whenever the user
# interacts with the application.
#
# Therefore, normal Python variables would be recreated
# during every rerun.
#
# st.session_state allows us to preserve data between
# Streamlit reruns.
#
# We create a "messages" list to store our chat history.
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# =========================================================
# DISPLAY PREVIOUS CHAT MESSAGES
# =========================================================

# Go through every message that we previously stored
# in session_state.
#
# Example:
#
# [
#     {
#         "role": "user",
#         "content": "Hello"
#     },
#     {
#         "role": "assistant",
#         "content": "Hi! How can I help?"
#     }
# ]
for message in st.session_state["messages"]:

    # Create a chat message container.
    #
    # message["role"] can be:
    #
    # "user"
    # "assistant"
    #
    # Streamlit uses the role to display the appropriate
    # chat bubble/avatar.
    with st.chat_message(message["role"]):

        # Display the actual message content.
        st.text(message["content"])


# =========================================================
# USER INPUT
# =========================================================

# Creates the chat input box at the bottom of the page.
#
# When the user enters a message and presses Enter,
# the entered text is stored in user_input.
#
# If the user hasn't entered anything,
# user_input will be None.
user_input = st.chat_input("Type here")


# =========================================================
# PROCESS USER INPUT
# =========================================================

# This block runs only when the user actually submits
# a message.
if user_input:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    # Add the user's message to our Streamlit chat history.
    #
    # We store:
    #   role    -> who sent the message
    #   content -> actual message
    #
    # Example:
    #
    # {
    #     "role": "user",
    #     "content": "What is LangGraph?"
    # }
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input
    })


    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    # Create a user chat bubble for the current message.
    with st.chat_message("user"):

        # Display what the user typed.
        st.text(user_input)


    # =====================================================
    # RUN CHATBOT WITH STREAMING
    # =====================================================

    # Create an assistant chat bubble.
    #
    # Everything generated inside this block will appear
    # inside the assistant's chat message.
    with st.chat_message("assistant"):

        # -------------------------------------------------
        # STREAM THE AI RESPONSE
        # -------------------------------------------------

        # chatbot.stream() does NOT wait for the entire
        # response before returning.
        #
        # Instead, it gives us chunks of the response
        # as they are generated.
        #
        # st.write_stream() receives those chunks and
        # displays them progressively in the UI.
        #
        # After the stream finishes, st.write_stream()
        # returns the complete generated text.
        #
        # Therefore, ai_message will contain the complete
        # AI response after streaming is finished.
        ai_message = st.write_stream(

            # For every message chunk produced by
            # chatbot.stream(), take only its content.
            #
            # message_chunk might conceptually look like:
            #
            # AIMessageChunk(content="Hello")
            #
            # We only want:
            #
            # "Hello"
            message_chunk.content

            # chatbot.stream() executes our LangGraph chatbot
            # in streaming mode.
            for message_chunk, metadata in chatbot.stream(

                # This is the input state we are giving
                # to our LangGraph chatbot.
                #
                # The user_input string is converted into
                # a LangChain HumanMessage object.
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                # Pass the LangGraph configuration.
                #
                # This contains our thread_id, which is used
                # to identify the conversation.
                config=config,

                # Tell LangGraph that we want to receive
                # message chunks while the chatbot is running.
                stream_mode="messages"
            )
        )


    # =====================================================
    # SAVE AI RESPONSE
    # =====================================================

    # At this point, the entire AI response has been
    # generated and displayed.
    #
    # st.write_stream() returned the complete response,
    # which is stored in ai_message.
    #
    # Now save it to Streamlit session state so that
    # it remains visible when Streamlit reruns.
    st.session_state["messages"].append({
        "role": "assistant",
        "content": ai_message
    })