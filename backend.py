import time

from datetime import datetime
from zoneinfo import ZoneInfo
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
import sqlite3
from langchain_community.document_loaders import PyPDFLoader
from groq import APIError, APIStatusError
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from tavily import TavilyClient, tavily
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import requests
from langgraph.types import interrupt, Command


load_dotenv()

llm = ChatGroq(model="qwen/qwen3.6-27b")


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


retriever = None  # module-level, updated whenever a document is (re)indexed


def _load_retriever_from_disk():
    """Load a retriever from an existing FAISS index on disk, if one exists."""
    global retriever
    INDEX_PATH = "faiss_index"

    if os.path.exists(INDEX_PATH):
        try:
            vector_store = FAISS.load_local(
                INDEX_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            )
        except Exception:
            retriever = None


def insert_rag_document(file_path):
    """Index a PDF into FAISS, merging into the existing index if one exists,
    and (re)build the module-level `retriever` so rag_tool can use it."""
    global retriever
    INDEX_PATH = "faiss_index"

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    if os.path.exists(INDEX_PATH):
        # Add to the existing index instead of overwriting it
        vector_store = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        vector_store.add_documents(chunks)
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

    vector_store.save_local(INDEX_PATH)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    return len(chunks)


# Load any previously-saved index at startup, so an uploaded PDF
# survives an app restart.
_load_retriever_from_disk()


@tool
def rag_tool(query: str)-> str:
    """
    Retrieve relevant information from the PDF document.

    Use this tool when the user asks factual or conceptual questions
    that may be answered using the stored PDF documents.

    Args:
        query: The question or search query used to retriever PDF content.
    """


    if retriever is None:
        return "No document has been uploaded yet. Please upload a PDF first"

    documents = retriever.invoke(query)

    if not documents:
        return "No relevant information was found in the PDF."

    formatted_documents = []

    for index, document in enumerate(documents,  start=1):
        source = document.metadata.get("source","Unknown source")
        page = document.metadata.get("page","Unknown page")

        formatted_documents.append(
            f"Document {index}\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Content: {document.page_content}"
        )

    return "\n\n".join(formatted_documents)


# Tools
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced"
)



@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.

    Use this tool whenever the user asks for arithmetic calculations.
    Example: '25 * 48' or '(100 + 50) / 5'
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Invalid mathematical expression."

@tool
def get_current_weather(city: str) -> str:
    """
    Get the current weather for a city.

    Input should be a city name, for example:
    'Delhi', 'London', or 'New York'.
    """

    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        return "OPENWEATHER_API_KEY is not configured."

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=10)

        if response.status_code != 200:
            return f"Could not get weather for {city}."

        data = response.json()

        temperature = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"]

        return (
            f"Weather in {city}: "
            f"{description}, "
            f"temperature {temperature}°C, "
            f"feels like {feels_like}°C, "
            f"humidity {humidity}%."
        )

    except requests.RequestException as e:
        return f"Weather service error: {str(e)}"

@tool
def web_lookup(query: str) -> str:
    """
    Search the web for general current information: news, events, facts.
    Do NOT use for stock prices or company financial updates — use get_stock_updates for those instead.
    """
    try:
        response = search_tool.invoke({"query": query})
        results = response.get("results", [])
        if not results:
            return "No relevant search results found."

        formatted_results = []
        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")[:500]
            url = result.get("url", "")
            formatted_results.append(f"Title: {title}\nContent: {content}\nURL: {url}")

        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def get_current_time(city: str) -> str:
    """
    Get the current local time for a city. Use this for ANY question about
    what time it is, current time, or the clock — never say you lack access to a live clock.
    """
    tz_map = {
        "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
        "london": "Europe/London", "new york": "America/New_York",
        "tokyo": "Asia/Tokyo", "paris": "Europe/Paris",
    }
    tz_name = tz_map.get(city.strip().lower())
    if not tz_name:
        return f"Timezone for {city} not known directly. Try web_lookup instead."
    now = datetime.now(ZoneInfo(tz_name))
    return f"Current time in {city}: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    
@tool
def get_stock_updates(company: str) -> str:
    """
    Get recent stock price movement, earnings, and market news for a SPECIFIC named company.
    This is the only tool to use for any stock/market/share-price question.
    Requires a company name — if the user hasn't named one, ask them which company before calling this.

    Example inputs: 'Apple', 'Tesla', 'NVIDIA'
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        return "TAVILY_API_KEY is not configured."

    try:
        tavily = TavilyClient(api_key=api_key)

        query = (
            f"{company} stock latest news today "
            f"stock price market update earnings"
        )

        response = tavily.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            topic="news"
        )

        results = response.get("results", [])

        if not results:
            return f"No recent stock updates found for {company}."

        formatted_results = []

        for result in results:
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")

            formatted_results.append(
                f"Title: {title}\n"
                f"Content: {content}\n"
                f"URL: {url}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Stock search error: {str(e)}"


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt and wait
    for a human decision ("yes" / anything else).

    IMPORTANT: This tool already returns a final, definitive status
    ("success" or "cancelled") for the requested purchase. Once it
    returns, the request is fully resolved — do NOT call this tool
    again for the same symbol/quantity. Simply report the returned
    status to the user in plain text.
    """

    # This pauses the graph and returns control to the caller
    decision = interrupt(
        f"Approve buying {quantity} shares of {symbol}? (yes/no)"
    )

    if isinstance(decision, str) and decision.lower() == "yes":
        return {
            "status": "success",
            "message": f"Purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }

    else:
        return {
            "status": "cancelled",
            "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity,
        }

# binding tools
tools = [ web_lookup, calculator, get_current_weather, get_stock_updates , get_current_time, rag_tool, purchase_stock]
llm_with_tools = llm.bind_tools(tools)
    

class ChatState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]



def sanitize_messages(messages):
    """Drop any tool-call/tool-response messages that lack a valid name,
    which break Harmony-based models like gpt-oss."""
    clean = []
    skip_tool_call_ids = set()

    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            bad_calls = [tc for tc in m.tool_calls if not tc.get("name")]
            if bad_calls:
                skip_tool_call_ids.update(tc.get("id") for tc in bad_calls)
                # drop the tool_calls entirely, keep any text content
                if m.content:
                    clean.append(AIMessage(content=m.content))
                continue
        if isinstance(m, ToolMessage):
            if not getattr(m, "name", None) or m.tool_call_id in skip_tool_call_ids:
                continue  # skip orphaned/malformed tool responses
        clean.append(m)

    return clean


SYSTEM_PROMPT = SystemMessage(content="""
You are a helpful Agentic Chatbot with access to several tools.

Tool usage instructions:

- Use `rag_tool` for questions about the uploaded PDF or document.
- Always retrieve relevant document content before answering PDF-related questions.
- Use `web_lookup` for current events, recent information, news, and general web searches.
- Use `calculator` for mathematical calculations. Do not calculate complex expressions manually when the calculator is available.
- Use `get_stock_updates` for stock prices, market updates, earnings, and company financial news.
- Use `get_current_weather` when the user asks about current weather for a location.
- Use `get_current_time` when the user asks for the current time in a city.
- Use `purchase_stock` when the user asks to buy/purchase shares of a stock. This tool
  already includes a human-approval step and returns a final "success" or "cancelled"
  status. Once it returns a result for a given symbol and quantity, that request is
  fully resolved: do NOT call `purchase_stock` again for the same request. Just tell
  the user the outcome in plain text. Only call it again if the user explicitly asks
  to buy shares again (a brand new request).

Answer general questions directly when no tool is required.

Do not invent tool names or parameters.
Use the tools only when they are appropriate for the user's request.

If the user asks about a PDF/document but no document has been uploaded or indexed,
tell the user that they need to upload a PDF first.
""")


def chat_node(state: ChatState):
    # Keep only recent messages to avoid sending an unnecessarily large history
    messages = [
        SYSTEM_PROMPT
    ] + sanitize_messages(state["messages"][-10:])

    response = llm_with_tools.invoke(messages)

    return {"messages":[response]}

 
tool_node = ToolNode(tools)


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)

graph = StateGraph(ChatState)

# add node
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

# add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools','chat_node')

chatbot = graph.compile(checkpointer=checkpoint)

def get_all_threads():
    all_threads = set()
    threads = checkpoint.list(None)
    for thread in threads:
        all_threads.add(thread.config['configurable']['thread_id'])

    return (list(all_threads))


def delete_thread(thread_id):
    checkpoint.delete_thread(thread_id)