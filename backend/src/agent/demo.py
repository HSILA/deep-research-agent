import asyncio
from typing import Dict, Any, List

import chainlit as cl
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agent.graph import generate_query, web_research, reflection, finalize_answer
from agent.state import OverallState, QueryGenerationState, WebSearchState


def _run_sync(func, *args, **kwargs):
    """Run a blocking function in a thread."""
    return asyncio.to_thread(func, *args, **kwargs)


async def generate_query_step(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    return await _run_sync(generate_query, state, config)


async def web_research_step(state: WebSearchState, config: RunnableConfig) -> OverallState:
    return await _run_sync(web_research, state, config)


async def reflection_step(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    return await _run_sync(reflection, state, config)


async def finalize_answer_step(state: OverallState, config: RunnableConfig) -> Dict[str, Any]:
    return await _run_sync(finalize_answer, state, config)


@cl.on_message
async def on_message(message: cl.Message):
    config = RunnableConfig()

    state: OverallState = {
        "messages": [HumanMessage(content=message.content)],
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "initial_search_query_count": 3,
        "max_research_loops": 1,
        "research_loop_count": 0,
        "reasoning_model": "gemini-2.5-flash-preview-04-17",
    }

    query_state = await generate_query_step(state, config)

    for idx, query in enumerate(query_state["query_list"]):
        research_update = await web_research_step({"search_query": query, "id": str(idx)}, config)
        for key, val in research_update.items():
            if isinstance(val, list):
                state.setdefault(key, [])
                state[key] += val
            else:
                state[key] = val

    reflection_update = await reflection_step(state, config)
    state.update(reflection_update)

    final = await finalize_answer_step(state, config)
    answer = final["messages"][-1].content

    await cl.Message(content=answer).send()
