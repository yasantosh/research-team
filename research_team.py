"""
AI Research Team - a multi-agent LangGraph project.

Flow:
    START -> planner -> searcher -> writer -> critic --(approved or max revisions)--> END
                                       ^                |
                                       +-- feedback ----+

Agents:
    planner  : breaks the topic into focused research questions
    searcher : runs a web search for each question and collects sources
    writer   : writes (or revises) a cited report from the sources
    critic   : checks the draft against the sources and approves or sends it back

Setup:
    pip install -U langgraph langchain langchain-anthropic pydantic ddgs
    export ANTHROPIC_API_KEY=...

Run:
    python research_team.py "Impact of Karpenter on Kubernetes cost optimization"

Change the model with the RESEARCH_MODEL env var, for example
"anthropic:claude-sonnet-5-5" (default) or "openai:gpt-4o" after installing
the matching langchain provider package.
"""

import argparse
import operator
import os
import uuid
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

MAX_REVISIONS = 3          # hard stop for the writer <-> critic loop
RESULTS_PER_QUESTION = 4   # search results kept for each research question

llm = init_chat_model(os.getenv("RESEARCH_MODEL", "anthropic:claude-sonnet-5-5"))


# --------------------------------------------------------------------------
# Shared state: every agent reads from and writes to this
# --------------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    questions: list[str]
    # operator.add is a reducer: updates from nodes are appended, not overwritten
    sources: Annotated[list[dict], operator.add]
    draft: str
    feedback: str
    approved: bool
    revisions: int


# Structured outputs keep the planner and critic predictable
class Plan(BaseModel):
    questions: list[str] = Field(description="3 to 5 focused research questions")


class Critique(BaseModel):
    approved: bool = Field(description="True only if the report is accurate, complete and well cited")
    feedback: str = Field(description="Specific, actionable fixes. Empty if approved.")


# --------------------------------------------------------------------------
# Tool: web search (swap this function for Tavily, Bedrock KB, etc.)
# --------------------------------------------------------------------------
def web_search(query: str, max_results: int = RESULTS_PER_QUESTION) -> list[dict]:
    try:
        from ddgs import DDGS
    except ImportError:  # older package name
        from duckduckgo_search import DDGS
    try:
        hits = DDGS().text(query, max_results=max_results)
    except Exception as exc:  # network errors should not crash the graph
        print(f"  [search failed for '{query}': {exc}]")
        return []
    return [
        {"question": query, "title": h.get("title", ""), "url": h.get("href", ""), "snippet": h.get("body", "")}
        for h in hits
    ]


def format_sources(sources: list[dict]) -> str:
    return "\n".join(
        f"[{i}] {s['title']} ({s['url']})\n    {s['snippet']}" for i, s in enumerate(sources, start=1)
    )


# --------------------------------------------------------------------------
# Nodes (agents)
# --------------------------------------------------------------------------
def planner(state: ResearchState) -> dict:
    plan = llm.with_structured_output(Plan).invoke(
        "You are a research planner. Break this topic into 3 to 5 focused, "
        f"searchable research questions.\n\nTopic: {state['topic']}"
    )
    print("PLANNER   questions:", *plan.questions, sep="\n  - ")
    return {"questions": plan.questions, "revisions": 0, "approved": False, "feedback": ""}


def searcher(state: ResearchState) -> dict:
    found: list[dict] = []
    for q in state["questions"]:
        found.extend(web_search(q))
    print(f"SEARCHER  collected {len(found)} sources")
    return {"sources": found}


def writer(state: ResearchState) -> dict:
    revision_note = ""
    if state.get("feedback"):
        revision_note = (
            "\n\nThis is a revision. Fix every point in the reviewer feedback below.\n"
            f"Reviewer feedback:\n{state['feedback']}\n\nPrevious draft:\n{state['draft']}"
        )
    prompt = (
        "You are a research writer. Write a clear, structured report in Markdown on the "
        "topic below. Use ONLY facts supported by the numbered sources, and cite them inline "
        "as [1], [2]. If sources are thin or conflicting, say so. End with a Sources section.\n\n"
        f"Topic: {state['topic']}\n\nSources:\n{format_sources(state['sources'])}{revision_note}"
    )
    draft = llm.invoke(prompt).content
    revisions = state["revisions"] + 1
    print(f"WRITER    draft #{revisions} written")
    return {"draft": draft, "revisions": revisions}


def critic(state: ResearchState) -> dict:
    review = llm.with_structured_output(Critique).invoke(
        "You are a strict research reviewer. Check the draft against the sources. "
        "Flag claims that are not supported by a cited source, missing citations, gaps "
        "versus the topic, and unclear writing. Approve only if the report is ready to publish.\n\n"
        f"Topic: {state['topic']}\n\nSources:\n{format_sources(state['sources'])}\n\n"
        f"Draft:\n{state['draft']}"
    )
    verdict = "APPROVED" if review.approved else "REVISE"
    print(f"CRITIC    {verdict}" + ("" if review.approved else f": {review.feedback[:120]}..."))
    return {"approved": review.approved, "feedback": review.feedback}


def route_after_critic(state: ResearchState) -> str:
    """Conditional edge: loop back to the writer unless approved or out of revisions."""
    if state["approved"] or state["revisions"] >= MAX_REVISIONS:
        return END
    return "writer"


# --------------------------------------------------------------------------
# Graph
# --------------------------------------------------------------------------
def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner)
    g.add_node("searcher", searcher)
    g.add_node("writer", writer)
    g.add_node("critic", critic)

    g.add_edge(START, "planner")
    g.add_edge("planner", "searcher")
    g.add_edge("searcher", "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", route_after_critic, {"writer": "writer", END: END})

    # MemorySaver checkpoints state per thread_id. Swap for SqliteSaver/PostgresSaver to persist.
    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent research team")
    parser.add_argument("topic", help="Topic to research")
    parser.add_argument("--out", default="report.md", help="Where to save the final report")
    args = parser.parse_args()

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # stream_mode="updates" yields after each node, so the prints above show progress
    for _ in graph.stream({"topic": args.topic}, config, stream_mode="updates"):
        pass

    final = graph.get_state(config).values
    status = "approved" if final["approved"] else f"stopped after {final['revisions']} revisions (not approved)"
    print(f"\nDone: {status}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(final["draft"])
    print(f"Report saved to {args.out}")


if __name__ == "__main__":
    main()
