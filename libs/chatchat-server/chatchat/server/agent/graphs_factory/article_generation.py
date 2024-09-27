import rich
from typing import Optional, Annotated, List, TypedDict, Any

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, add_messages, END
from langgraph.graph.graph import CompiledGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from chatchat.server.utils import build_logger, get_st_graph_memory, get_tool
from .graphs_registry import (
    regist_graph,
    InputHandler,
    EventHandler,
)

logger = build_logger()


class ArticleListResponse(BaseModel):
    """Respond to the user with this"""
    article_list: List[str] = Field(description="The list of article")


class ArticleGenerationState(TypedDict):
    """
    定义一个基础 State 供 各类 graph 继承, 其中:
    1. messages 为所有 graph 的核心信息队列, 所有聊天工作流均应该将关键信息补充到此队列中;
    2. history 为所有工作流单次启动时获取 history_len 的 messages 所用(节约成本, 及防止单轮对话 tokens 占用长度达到 llm 支持上限),
    history 中的信息理应是可以被丢弃的.
    """
    messages: Annotated[List[BaseMessage], add_messages]
    article_links: Optional[str]
    image_links: Optional[str]
    article_links_list: Optional[list[str]]
    image_links_list: Optional[list[str]]
    article_list: Optional[list[str]]
    llm: Optional[str]
    article: Optional[str]
    user_feedback: Optional[str]


# 用来暂停 langgraph, 使得用户传入 待爬文章链接 和 图片链接
async def article_generation_init_break_point(state: ArticleGenerationState) -> ArticleGenerationState:
    logger.info("This is the article_generation_INIT_break_point node and now i will break this graph.")
    return state


# 用来暂停 langgraph
async def article_generation_repeat_break_point(state: ArticleGenerationState) -> ArticleGenerationState:
    logger.info("This is the article_generation_REPEAT_break_point node and now i will break this graph.")
    return state


# 用户传入待爬文章链接后的处理
async def url_handler(state: ArticleGenerationState) -> ArticleGenerationState:
    # 处理用户输入的文章和图片链接
    article_links_list = [link.strip() for link in state["article_links"].split('\n') if link.strip()]
    image_links_list = [link.strip() for link in state["image_links"].split('\n') if link.strip()]
    state["article_links_list"] = article_links_list
    state["image_links_list"] = image_links_list

    call_spider_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a web crawler AI designed to efficiently extract article content from specified web links using the crawler tool (url_reader).

Instructions:
1. Analyze the provided list of article links and determine if the crawler tool needs to be called once for a single link or multiple times for several links.
2. Crawl the articles from the following links:
{article_links_list}

Requirements:
- Identify and remove any advertising-related content, promotional material, or unrelated sections from the articles.
- Return only the main content of each article without summarizing, rewriting, or altering the text.
- Ensure that the final output is a list of strings, where each string represents the complete content of an individual article.

Additional Considerations:
- If a link leads to a page that cannot be crawled or does not contain valid article content, return a placeholder string indicating the issue (e.g., "Unable to retrieve content from [link]").
- Maintain the order of the links in the output list to correspond with the input list.

Final Output:
- A list of strings containing the cleaned article content, with each element corresponding to the respective link in the input list.
"""
            ),
        ]
    )
    prompt = call_spider_template.format(article_links_list=state["article_links_list"])
    state["messages"] = [HumanMessage(content=prompt)]

    return state


# 用来爬取用户输入的网页文章内容
def get_spider_graph(llm: ChatOpenAI, memory: Any) -> CompiledStateGraph:
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be an instance of ChatOpenAI")

    url_reader = get_tool(name="url_reader")
    tools = [url_reader]
    llm_with_tools = llm.bind_tools(tools)

#     respond_node_template = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 """
# Requirements:
# - Return only the main content of each article as it is, without summarizing, rewriting, or altering the text in any way.
# - The output should not include any additional commentary, explanations, or formatting changes.
# - Ensure that the final output is a list of strings, where each string represents the complete content of an individual article from the input list.
#
# Final Output:
# - A list of strings containing the full content of each article. Each element in the list should correspond to one article from the input, preserving the original text exactly as it appears.
# """
#             ),
#         ]
#     )
    # rich.print(respond_node_template)
    # llm_with_structured_output = respond_node_template | llm.with_structured_output(ArticleListResponse)
    # rich.print(llm_with_structured_output)

    async def function_call(state: ArticleGenerationState) -> ArticleGenerationState:
        llm_result = llm_with_tools.invoke(state["messages"])
        state["messages"] = [llm_result]
        print("--- state ---")
        rich.print(state)
        return state

    # # Define the function that responds to the user
    # async def respond(state: ArticleGenerationState) -> ArticleGenerationState:
    #     # We call the model with structured output in order to return the same format to the user every time
    #     # state['messages'][-2] is the last ToolMessage in the convo, which we convert to a HumanMessage for the model to use
    #     # We could also pass the entire chat history, but this saves tokens since all we care to structure is the output of the tool
    #     message = llm_with_structured_output.invoke([HumanMessage(content=state['messages'][-2].content)])
    #     rich.print(message)
    #     state["article_list"] = message["article_list"]
    #     # We return the final answer
    #     return state

    # # Define the function that determines whether to continue or not
    # async def should_continue(state: ArticleGenerationState):
    #     messages = state["messages"]
    #     last_message = messages[-1]
    #     # If there is no function call, then we respond to the user
    #     if not last_message.tool_calls:
    #         return "respond"
    #     # Otherwise if there is, we continue
    #     else:
    #         return "continue"

    graph_builder = StateGraph(ArticleGenerationState)

    tool_node = ToolNode(tools=tools)

    graph_builder.add_node("function_call", function_call)
    # graph_builder.add_node("respond", respond)
    graph_builder.add_node("tools", tool_node)

    # Set the entrypoint as `agent`
    # This means that this node is the first one called
    graph_builder.set_entry_point("function_call")

    # We now add a conditional edge
    # graph_builder.add_conditional_edges(
    #     "function_call",
    #     should_continue,
    #     {
    #         "continue": "tools",
    #         "respond": "respond",
    #     },
    # )

    graph_builder.add_conditional_edges(
        "function_call",
        tools_condition,
    )

    graph_builder.add_edge("tools", "function_call")
    # graph_builder.add_edge("respond", END)

    graph = graph_builder.compile(checkpointer=memory)

    return graph


class ArticleGenerationEventHandler(EventHandler):
    def __init__(self):
        pass

    def handle_event(self, node: str, events: ArticleGenerationState) -> BaseMessage:
        """
        event example:
        {
            'messages': [HumanMessage(
                            content='The youtube video of Xiao Yixian in Fights Break Sphere?',
                            id='b9c5468a-7340-425b-ae6f-2f584a961014')],
            'history': [HumanMessage(
                            content='The youtube video of Xiao Yixian in Fights Break Sphere?',
                            id='b9c5468a-7340-425b-ae6f-2f584a961014')]
        }
        """
        return events["messages"][0]


@regist_graph(name="article_generation",
              input_handler=InputHandler,
              event_handler=ArticleGenerationEventHandler)
def article_generation(llm: ChatOpenAI, tools: list[BaseTool], history_len: int) -> CompiledGraph:
    """
    description: https://langchain-ai.github.io/langgraph/tutorials/introduction/
    """
    if not isinstance(llm, ChatOpenAI):
        raise TypeError("llm must be an instance of ChatOpenAI")
    if not all(isinstance(tool, BaseTool) for tool in tools):
        raise TypeError("All items in tools must be instances of BaseTool")

    memory = get_st_graph_memory()
    rich.print(memory)

    graph_builder = StateGraph(ArticleGenerationState)

    llm_with_tools = llm.bind_tools(tools)

    async def agent(state: ArticleGenerationState) -> ArticleGenerationState:
        messages = llm_with_tools.invoke(state["messages"])
        state["messages"] = [messages]
        return state

    tool_node = ToolNode(tools=tools)
    spider_graph = get_spider_graph(llm=llm, memory=memory)

    graph_builder.add_node("agent", agent)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("article_generation_init_break_point", article_generation_init_break_point)
    graph_builder.add_node("url_handler", url_handler)
    graph_builder.add_node("spider", spider_graph)

    graph_builder.set_entry_point("article_generation_init_break_point")
    graph_builder.add_edge("article_generation_init_break_point", "url_handler")
    graph_builder.add_edge("url_handler", "spider")
    graph_builder.add_edge("spider", "agent")
    graph_builder.add_conditional_edges(
        "agent",
        tools_condition,
    )
    graph_builder.add_edge("tools", "agent")

    graph = graph_builder.compile(checkpointer=memory, interrupt_after=["article_generation_init_break_point"])

    return graph
