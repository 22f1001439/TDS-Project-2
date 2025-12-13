from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.messages import trim_messages, HumanMessage

from typing import TypedDict, Annotated, List
import os
import time
from dotenv import load_dotenv

from shared_store import url_time
from tools import (
    get_rendered_html,
    download_file,
    post_request,
    run_code,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
)

# -------------------------------------------------
# ENV
# -------------------------------------------------
load_dotenv()

EMAIL = os.getenv("EMAIL")
SECRET = os.getenv("SECRET")

RECURSION_LIMIT = 5000
MAX_TOKENS = 60000

# -------------------------------------------------
# STATE
# -------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List, add_messages]

TOOLS = [
    run_code,
    get_rendered_html,
    download_file,
    post_request,
    add_dependencies,
    ocr_image_tool,
    transcribe_audio,
    encode_image_to_base64,
]

# -------------------------------------------------
# LLM INIT (Gemini via AI Pipe)
# -------------------------------------------------
rate_limiter = InMemoryRateLimiter(
    requests_per_second=4 / 60,
    check_every_n_seconds=1,
    max_bucket_size=4,
)

llm = ChatOpenAI(
    model="openai/gpt-4.1",
    temperature=0,
    max_tokens=MAX_TOKENS,
    api_key=os.getenv("OPENAI_API_KEY"),      # AI Pipe token
    base_url=os.getenv("OPENAI_BASE_URL"),    # AI Pipe base URL
    rate_limiter=rate_limiter,
).bind_tools(TOOLS)

# -------------------------------------------------
# SYSTEM PROMPT
# -------------------------------------------------
SYSTEM_PROMPT = f"""
You are an autonomous quiz-solving agent.

Your job is to:
1. Load each quiz page from the given URL.
2. Extract instructions, parameters, and submit endpoint.
3. Solve tasks exactly.
4. Submit answers ONLY to the correct endpoint.
5. Follow new URLs until none remain, then output END.

Rules:
- For base64 generation of an image NEVER use your own code, always use the "encode_image_to_base64" tool.
- Never hallucinate URLs or fields.
- Never shorten endpoints.
- Always inspect server responses.
- Never stop early.
- Use tools whenever needed.

Always include:
email = {EMAIL}
secret = {SECRET}
"""

# -------------------------------------------------
# MALFORMED JSON HANDLER
# -------------------------------------------------
def handle_malformed_node(state: AgentState):
    print("--- MALFORMED TOOL CALL DETECTED ---")
    return {
        "messages": [
            {
                "role": "user",
                "content": (
                    "SYSTEM ERROR: Your last tool call was malformed JSON. "
                    "Please retry with valid JSON only."
                ),
            }
        ]
    }

# -------------------------------------------------
# AGENT NODE
# -------------------------------------------------
def agent_node(state: AgentState):
    cur_time = time.time()
    cur_url = os.getenv("url")
    prev_time = url_time.get(cur_url)
    offset = os.getenv("offset", "0")

    # ---- TIMEOUT HANDLING ----
    if prev_time is not None:
        diff = cur_time - float(prev_time)
        if diff >= 180 or (offset != "0" and cur_time - float(offset) > 90):
            fail_msg = HumanMessage(
                content=(
                    "You have exceeded the time limit. "
                    "Immediately submit a WRONG answer using the post_request tool."
                )
            )
            result = llm.invoke(state["messages"] + [fail_msg])
            return {"messages": [result]}

    # ---- NO TOKEN TRIMMING (GPT-4.1 SAFE) ----
    trimmed_messages = state["messages"]

    # ---- ENSURE HUMAN MESSAGE EXISTS ----
    if not any(m.type == "human" for m in trimmed_messages):
        trimmed_messages.append(
            HumanMessage(
                content=f"Context trimmed. Continue processing URL: {cur_url}"
            )
        )

    # ---- INVOKE MODEL ----
    result = llm.invoke(trimmed_messages)
    return {"messages": [result]}

# -------------------------------------------------
# ROUTER
# -------------------------------------------------
def route(state: AgentState):
    last = state["messages"][-1]

    if "finish_reason" in last.response_metadata:
        if last.response_metadata["finish_reason"] == "MALFORMED_FUNCTION_CALL":
            return "handle_malformed"

    if getattr(last, "tool_calls", None):
        return "tools"

    content = getattr(last, "content", None)
    if isinstance(content, str) and content.strip() == "END":
        return END

    if isinstance(content, list) and content and content[0].get("text") == "END":
        return END

    return "agent"

# -------------------------------------------------
# GRAPH
# -------------------------------------------------
graph = StateGraph(AgentState)

graph.add_node("agent", agent_node)
graph.add_node("tools", ToolNode(TOOLS))
graph.add_node("handle_malformed", handle_malformed_node)

graph.add_edge(START, "agent")
graph.add_edge("tools", "agent")
graph.add_edge("handle_malformed", "agent")

graph.add_conditional_edges(
    "agent",
    route,
    {
        "tools": "tools",
        "agent": "agent",
        "handle_malformed": "handle_malformed",
        END: END,
    },
)

app = graph.compile()

# -------------------------------------------------
# RUNNER
# -------------------------------------------------
def run_agent(url: str):
    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": url},
    ]

    app.invoke(
        {"messages": initial_messages},
        config={"recursion_limit": RECURSION_LIMIT},
    )

    print("Tasks completed successfully!")
