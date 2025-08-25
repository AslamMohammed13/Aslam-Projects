from typing import TypedDict, Annotated, Optional
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# Initialize memory saver for checkpointing
memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]

# ---------------------------
# TOOLS
# ---------------------------

# 1. Tavily Web Search Tool
search_tool = TavilySearchResults(max_results=4)

# 2. RAG Retriever Tool (Chroma DB)
pdf_path = "tj_test.pdf"  # replace with your file path
loader = PyPDFLoader(pdf_path)
pages = loader.load()  # One Document per page

# Step 2: Split into smaller chunks (better retrieval accuracy)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # tokens/characters depending on tokenizer
    chunk_overlap=100
)
docs = text_splitter.split_documents(pages)

# Step 3: Create high-quality embeddings (text-embedding-3-large)
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

# Step 4: Store in Chroma vector database
db = Chroma.from_documents(docs, embedding_function)
print(f"Loaded {len(docs)} document chunks from PDF and stored in Chroma.")

# Step 5: Create retriever with similarity search and k=15
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 15}  # you can change to 10 or 20
)

@tool("vector_db_retrieval", return_direct=False)
async def vector_db_retrieval(query: str):
    """Retrieve relevant documents from internal knowledge base (Chroma DB)."""
    docs = retriever.get_relevant_documents(query)
    return [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]

# Register tools
tools = [search_tool, vector_db_retrieval]

# LLM with tool binding
llm = ChatOpenAI(model="gpt-4o")
llm_with_tools = llm.bind_tools(tools=tools)

# ---------------------------
# GRAPH NODES
# ---------------------------

async def model(state: State):
    result = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [result]}

async def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tool_node"
    else:
        return END

async def tool_node(state):
    """Custom tool node that handles tool calls from the LLM."""
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if tool_name == "tavily_search_results_json":
            search_results = await search_tool.ainvoke(tool_args)
            tool_message = ToolMessage(
                content=str(search_results), tool_call_id=tool_id, name=tool_name
            )
            tool_messages.append(tool_message)

        elif tool_name == "vector_db_retrieval":
            query = tool_args.get("query", "")
            results = await vector_db_retrieval.ainvoke({"query": query})
            tool_message = ToolMessage(
                content=str(results), tool_call_id=tool_id, name=tool_name
            )
            tool_messages.append(tool_message)

    return {"messages": tool_messages}

# ---------------------------
# GRAPH DEFINITION
# ---------------------------

graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")
graph = graph_builder.compile(checkpointer=memory)

# ---------------------------
# FASTAPI APP
# ---------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

def serialise_ai_message_chunk(chunk):
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    else:
        raise TypeError(
            f"Object of type {type(chunk).__name__} is not correctly formatted for serialisation"
        )

async def generate_chat_responses(message: str, checkpoint_id: Optional[str] = None):
    is_new_conversation = checkpoint_id is None
    
    if is_new_conversation:
        new_checkpoint_id = str(uuid4())
        config = {"configurable": {"thread_id": new_checkpoint_id}}
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]}, version="v2", config=config
        )
        yield f"data: {{\"type\": \"checkpoint\", \"checkpoint_id\": \"{new_checkpoint_id}\"}}\n\n"
    else:
        config = {"configurable": {"thread_id": checkpoint_id}}
        events = graph.astream_events(
            {"messages": [HumanMessage(content=message)]}, version="v2", config=config
        )

    async for event in events:
        event_type = event["event"]

        if event_type == "on_chat_model_stream":
            chunk_content = serialise_ai_message_chunk(event["data"]["chunk"])
            safe_content = chunk_content.replace("'", "\\'").replace("\n", "\\n")
            yield f"data: {{\"type\": \"content\", \"content\": \"{safe_content}\"}}\n\n"

        elif event_type == "on_chat_model_end":
            tool_calls = event["data"]["output"].tool_calls if hasattr(event["data"]["output"], "tool_calls") else []
            
            if tool_calls:
                if any(call["name"] == "tavily_search_results_json" for call in tool_calls):
                    search_query = tool_calls[0]["args"].get("query", "")
                    safe_query = search_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                    yield f"data: {{\"type\": \"search_start\", \"query\": \"{safe_query}\"}}\n\n"
                
                if any(call["name"] == "vector_db_retrieval" for call in tool_calls):
                    retrieval_query = tool_calls[0]["args"].get("query", "")
                    safe_query = retrieval_query.replace('"', '\\"').replace("'", "\\'").replace("\n", "\\n")
                    yield f"data: {{\"type\": \"retrieval_start\", \"query\": \"{safe_query}\"}}\n\n"

        elif event_type == "on_tool_end" and event["name"] == "tavily_search_results_json":
            output = event["data"]["output"]
            if isinstance(output, list):
                urls = [item["url"] for item in output if isinstance(item, dict) and "url" in item]
                urls_json = json.dumps(urls)
                yield f"data: {{\"type\": \"search_results\", \"urls\": {urls_json}}}\n\n"

        elif event_type == "on_tool_end" and event["name"] == "vector_db_retrieval":
            output = event["data"]["output"]
            if isinstance(output, list):
                docs_json = json.dumps(output)
                yield f"data: {{\"type\": \"retrieval_results\", \"docs\": {docs_json}}}\n\n"

    yield f"data: {{\"type\": \"end\"}}\n\n"

@app.get("/chat_stream/{message}")
async def chat_stream(message: str, checkpoint_id: Optional[str] = Query(None)):
    return StreamingResponse(
        generate_chat_responses(message, checkpoint_id), media_type="text/event-stream"
    )
