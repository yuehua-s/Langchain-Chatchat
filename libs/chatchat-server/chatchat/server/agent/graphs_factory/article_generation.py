from typing import Optional

from langchain_openai.chat_models import ChatOpenAI
from langchain_core.tools import BaseTool
from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode, tools_condition

from chatchat.server.utils import build_logger, get_st_graph_memory
from .graphs_registry import (
    regist_graph,
    InputHandler,
    EventHandler,
    State,
    async_history_manager,
)

logger = build_logger()


class ArticleGenerationState(State):
    """
    定义一个基础 State 供 各类 graph 继承, 其中:
    1. messages 为所有 graph 的核心信息队列, 所有聊天工作流均应该将关键信息补充到此队列中;
    2. history 为所有工作流单次启动时获取 history_len 的 messages 所用(节约成本, 及防止单轮对话 tokens 占用长度达到 llm 支持上限),
    history 中的信息理应是可以被丢弃的.
    """
    article_links: Optional[str]
    image_links: Optional[str]
    llm: Optional[str]
    article: Optional[str]
    user_feedback: Optional[str]


# 用来暂停 langgraph
async def article_generation_init_break_point(state: ArticleGenerationState) -> ArticleGenerationState:
    logger.info("This is the article_generation_INIT_break_point node and now i will break this graph.")
    return state


# 用来暂停 langgraph
async def article_generation_repeat_break_point(state: ArticleGenerationState) -> ArticleGenerationState:
    logger.info("This is the article_generation_REPEAT_break_point node and now i will break this graph.")
    return state


# 获取用户反馈后的处理
async def human_feedback(state: ArticleGenerationState) -> ArticleGenerationState:
    # 这里可以添加逻辑来处理用户反馈
    # 例如，等待用户输入并更新 state["user_feedback"]
    logger.info("this is the human_feedback node.")
    # 处理用户输入的链接
    # article_links_list = [link.strip() for link in st.session_state["article_links"].split('\n') if link.strip()]
    # image_links_list = [link.strip() for link in st.session_state["image_links"].split('\n') if link.strip()]
    # st.session_state["article_links_list"] = article_links_list
    # st.session_state["image_links_list"] = image_links_list
    # article_links_list = st.session_state["article_links_list"]
    # image_links_list = st.session_state["image_links_list"]
    # logger.info(f"当前文章链接-处理后: {article_links_list}")
    # logger.info(f"当前图片链接-处理后: {image_links_list}")
    import rich
    rich.print(state)
    return state


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

    import rich
    memory = get_st_graph_memory()
    rich.print(memory)

    graph_builder = StateGraph(ArticleGenerationState)

    llm_with_tools = llm.bind_tools(tools)

    async def history_manager(state: ArticleGenerationState) -> ArticleGenerationState:
        state = await async_history_manager(state, history_len)
        return state

    async def chatbot(state: ArticleGenerationState) -> ArticleGenerationState:
        # ToolNode 默认只将结果追加到 messages 队列中, 所以需要手动在 history 中追加 ToolMessage 结果, 否则报错如下:
        # Error code: 400 -
        # {
        #     "error": {
        #         "message": "Invalid parameter: messages with role 'tool' must be a response to a preceeding message with 'tool_calls'.",
        #         "type": "invalid_request_error",
        #         "param": "messages.[1].role",
        #         "code": null
        #     }
        # }
        if isinstance(state["messages"][-1], ToolMessage):
            state["history"].append(state["messages"][-1])

        messages = llm_with_tools.invoke(state["history"])
        state["messages"] = [messages]
        # 因为 chatbot 执行依赖于 state["history"], 所以在同一次 workflow 没有执行结束前, 需要将每一次输出内容都追加到 state["history"] 队列中缓存起来
        state["history"].append(messages)
        return state

    tool_node = ToolNode(tools=tools)

    graph_builder.add_node("history_manager", history_manager)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("article_generation_init_break_point", article_generation_init_break_point)
    graph_builder.add_node("human_feedback", human_feedback)

    graph_builder.set_entry_point("history_manager")
    graph_builder.add_edge("history_manager", "article_generation_init_break_point")
    graph_builder.add_edge("article_generation_init_break_point", "human_feedback")
    graph_builder.add_edge("human_feedback", "chatbot")
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
    )
    graph_builder.add_edge("tools", "chatbot")

    graph = graph_builder.compile(checkpointer=memory, interrupt_after=["article_generation_init_break_point"])

    return graph
