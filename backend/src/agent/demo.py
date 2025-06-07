"""Chainlit demo app for the research agent."""
import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from .graph import (
    evaluate_research,
    finalize_answer,
    generate_query,
    reflection,
    web_research,
)
from .state import OverallState

# Wrap existing functions with Chainlit step decorators for nice UI tracking
generate_query_step = cl.step(name="Generate Queries", type="llm")(generate_query)
web_research_step = cl.step(name="Web Research", type="tool")(web_research)
reflection_step = cl.step(name="Reflection", type="llm")(reflection)
finalize_answer_step = cl.step(name="Finalize Answer", type="llm")(finalize_answer)


@cl.on_message
async def main(message: cl.Message) -> None:
    """Run the research agent for the incoming user question."""
    question = message.content
    state: OverallState = {
        "messages": [HumanMessage(content=question)],
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": 0,
        "max_research_loops": 0,
        "research_loop_count": 0,
        "reasoning_model": "",
    }

    config = RunnableConfig(configurable={})

    # 1. Generate initial search queries
    query_state = await cl.make_async(generate_query_step)(state, config)
    state.update(query_state)

    # 2. Perform web research for each generated query
    for idx, query in enumerate(state["query_list"]):
        search_state = await cl.make_async(web_research_step)(
            {"search_query": query["query"], "id": str(idx)},
            config,
        )
        state["search_query"] += search_state["search_query"]
        state["web_research_result"] += search_state["web_research_result"]
        state["sources_gathered"] += search_state["sources_gathered"]

    # 3. Reflection loop to evaluate and possibly continue searching
    while True:
        reflection_state = await cl.make_async(reflection_step)(state, config)
        state.update(reflection_state)

        decision = evaluate_research(reflection_state, config)
        if decision == "finalize_answer":
            break
        for send in decision:
            data = send.arg
            loop_state = await cl.make_async(web_research_step)(data, config)
            state["search_query"] += loop_state["search_query"]
            state["web_research_result"] += loop_state["web_research_result"]
            state["sources_gathered"] += loop_state["sources_gathered"]

    # 4. Finalize the answer and respond to the user
    final_state = await cl.make_async(finalize_answer_step)(state, config)
    state.update(final_state)
    final_answer = state["messages"][-1].content
    await cl.Message(content=final_answer).send()
