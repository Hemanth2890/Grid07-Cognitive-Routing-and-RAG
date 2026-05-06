import os
import json
from typing import TypedDict, Optional
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Load environment variables (GROQ_API_KEY)
load_dotenv()

# Reuse personas from Phase 1
from router import BOT_PERSONAS


# The mock search tool
@tool
def mock_searxng_search(query: str) -> str:
    """
    Mock web search tool. Returns hardcoded recent-looking headlines
    based on keywords in the query. Simulates a real SearxNG / web
    search integration without needing API access.
    """
    query_lower = query.lower()

    # Keyword-based mock results
    headlines = []
    if any(k in query_lower for k in ["crypto", "bitcoin", "btc", "ethereum"]):
        headlines.append("Bitcoin hits new all-time high amid regulatory ETF approvals.")
        headlines.append("BlackRock files for new Ethereum spot ETF.")
    if any(k in query_lower for k in ["ai", "openai", "llm", "chatgpt", "gpt"]):
        headlines.append("OpenAI releases GPT-5 with autonomous agent capabilities.")
        headlines.append("Anthropic raises $5B at $60B valuation as AI race heats up.")
    if any(k in query_lower for k in ["fed", "interest", "rate", "inflation", "market"]):
        headlines.append("Fed signals two more rate cuts in 2026 amid cooling inflation.")
        headlines.append("S&P 500 closes at record high as tech stocks surge.")
    if any(k in query_lower for k in ["privacy", "surveillance", "big tech", "data"]):
        headlines.append("Leaked memo reveals Meta tracked users across 200+ apps without consent.")
        headlines.append("EU passes strictest data protection law since GDPR.")
    if any(k in query_lower for k in ["elon", "musk", "tesla", "spacex"]):
        headlines.append("SpaceX successfully lands Starship on Mars simulation pad.")
        headlines.append("Tesla unveils fully autonomous robotaxi fleet in Austin.")
    if any(k in query_lower for k in ["climate", "environment", "warming"]):
        headlines.append("2025 confirmed as hottest year on record, scientists warn.")

    if not headlines:
        headlines = [
            "Tech industry layoffs continue as automation accelerates.",
            "Government proposes new AI regulation framework.",
        ]

    return "\n".join(f"- {h}" for h in headlines)


# Structured output schema (Pydantic enforces guaranteed JSON shape)
class BotPost(BaseModel):
    """The final post output. Schema enforced via Groq's structured output."""
    bot_id: str = Field(description="The unique ID of the bot that wrote the post.")
    topic: str = Field(description="The high-level topic of the post (2-4 words).")
    post_content: str = Field(
        description="The post itself. Must be under 280 characters, opinionated, in the bot's voice."
    )


# LangGraph state definition - flows between nodes
class GraphState(TypedDict):
    """State that flows through the graph between nodes."""
    bot_id: str
    persona: str
    search_query: Optional[str]
    search_results: Optional[str]
    final_post: Optional[dict]


# LLM setup
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,  # some creativity for personality
)


# Node 1: Decide Search
def decide_search_node(state: GraphState) -> GraphState:
    """LLM picks a topic and crafts a search query based on persona."""
    print(f"\n[Node 1: Decide Search] Bot '{state['bot_id']}' is thinking...")

    system_prompt = (
        "You are an AI bot deciding what to post about today. "
        "Based on your persona, pick ONE topic you genuinely care about right now "
        "and craft a SHORT search query (3-6 words) to gather context. "
        "Return ONLY the search query, no explanation, no quotes."
    )
    human_prompt = f"My persona:\n{state['persona']}\n\nWhat do I want to search for?"

    response = llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content=human_prompt),])

    query = response.content.strip().strip('"').strip("'")
    print(f"   Search query chosen: {query}")

    return {**state, "search_query": query}


# Node 2: Web Search
def web_search_node(state: GraphState) -> GraphState:
    """Calls the mock search tool."""
    print(f"[Node 2: Web Search] Searching: '{state['search_query']}'")

    results = mock_searxng_search.invoke({"query": state["search_query"]})
    print(f"   Headlines found:\n{results}")

    return {**state, "search_results": results}


# Node 3: Draft Post
def draft_post_node(state: GraphState) -> GraphState:
    """LLM drafts the final post in strict JSON format."""

    # Bind the Pydantic schema to force structured output
    structured_llm = llm.with_structured_output(BotPost)

    system_prompt = (
        f"You are bot '{state['bot_id']}' with this persona:\n{state['persona']}\n\n"
        "Write a SHORT, opinionated post (under 280 characters) in your voice. "
        "Use the search results as factual grounding but inject your personality. "
        "Be punchy, controversial, in-character. NO hashtags, NO emojis."
    )
    human_prompt = (
        f"Recent headlines for context:\n{state['search_results']}\n\n"
        f"Now write your post about this topic. Set bot_id to '{state['bot_id']}'."
    )

    result: BotPost = structured_llm.invoke([SystemMessage(content=system_prompt),HumanMessage(content=human_prompt),])

    # Convert Pydantic model to dict for the state
    post_dict = result.model_dump()
    print(f"   Post drafted.")

    return {**state, "final_post": post_dict}


# Build the LangGraph workflow
def build_content_engine():
    """Wires the three nodes into a linear state machine."""
    workflow = StateGraph(GraphState)

    workflow.add_node("decide_search", decide_search_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("draft_post", draft_post_node)

    workflow.set_entry_point("decide_search")
    workflow.add_edge("decide_search", "web_search")
    workflow.add_edge("web_search", "draft_post")
    workflow.add_edge("draft_post", END)

    return workflow.compile()


# Run the engine for every bot when executed directly
if __name__ == "__main__":
    print("AUTONOMOUS CONTENT ENGIN")

    engine = build_content_engine()

    for bot_id, persona in BOT_PERSONAS.items():
        print(f"RUNNING ENGINE FOR: {bot_id}")

        initial_state: GraphState = {
            "bot_id": bot_id,
            "persona": persona,
            "search_query": None,
            "search_results": None,
            "final_post": None,
        }

        final_state = engine.invoke(initial_state)

        print(f"\nFINAL OUTPUT (strict JSON):")
        print(json.dumps(final_state["final_post"], indent=2))
        print(f"   Length: {len(final_state['final_post']['post_content'])} chars")