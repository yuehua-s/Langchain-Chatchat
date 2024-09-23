import uuid
import asyncio

import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from streamlit_extras.bottom_container import bottom

from chatchat.settings import Settings
from chatchat.server.chat.graph_chat import create_agent_models
from chatchat.webui_pages.utils import *
from chatchat.webui_pages.dialogue.dialogue import list_graphs, list_tools
from chatchat.server.utils import (
    build_logger,
    get_config_models,
    get_config_platforms,
    get_default_llm,
    get_history_len,
    get_recursion_limit,
    get_graph_instance,
    get_tool, get_graph_memory,
)

logger = build_logger()


def init_conversation_id():
    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = str(uuid.uuid4())


def extract_node_and_response(data):
    # 获取第一个键值对，作为 node
    if not data:
        raise ValueError("数据为空")

    # 获取第一个键及其对应的值
    node = next(iter(data))
    response = data[node]

    return node, response


async def handle_user_input(
        graph_input: Any,
        graph: CompiledStateGraph,
        graph_config: Dict,
):
    events = graph.astream(input=graph_input, config=graph_config, stream_mode="updates")
    if events:
        async for event in events:
            node, response = extract_node_and_response(event)
            if node == "history_manager":  # history_manager node 为内部实现, 不外显
                continue
            if node == "article_generation_init_break_point":
                with st.chat_message("assistant"):
                    st.write("请进行初始化设置")
                    st.session_state.messages.append({"role": "assistant", "content": "请进行初始化"})
                article_generation_init_setting()
                continue
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.status(node, expanded=True) as status:
                    st.json(response, expanded=True)
                    status.update(
                        label=node, state="complete", expanded=False
                    )
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": event})


@st.dialog("模型配置", width="large")
def llm_model_setting():
    cols = st.columns(3)
    platforms = ["所有"] + list(get_config_platforms())
    platform = cols[0].selectbox("模型平台设置(Platform)", platforms)
    llm_models = list(
        get_config_models(
            model_type="llm", platform_name=None if platform == "所有" else platform
        )
    )
    llm_model = cols[1].selectbox("模型设置(LLM)", llm_models)
    temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=st.session_state["temperature"])

    if st.button("确认"):
        st.session_state["platform"] = platform
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.rerun()


@st.dialog("输入链接", width="large")
def article_generation_init_setting():
    article_links = st.text_area("文章链接")
    image_links = st.text_area("图片链接")

    if st.button("确认"):
        st.session_state["article_links"] = article_links
        st.session_state["image_links"] = image_links
        st.session_state["article_generation_init_break_point"] = True

        user_input = (f"文章链接: {article_links}\n"
                      f"图片链接: {image_links}")

        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        st.rerun()


@st.dialog("文章重写确认", width="large")
def article_generation_repeat_setting():
    cols = st.columns(3)
    platforms = ["所有"] + list(get_config_platforms())
    platform = cols[0].selectbox("模型平台设置(Platform)", platforms)
    llm_models = list(
        get_config_models(
            model_type="llm", platform_name=None if platform == "所有" else platform
        )
    )
    llm_model = cols[1].selectbox("模型设置(LLM)", llm_models)
    temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=st.session_state["temperature"])
    article = st.write("当前的文章内容如下: xxxxxxxxxx")
    prompt = st.text_area("指令(Prompt):", value=st.session_state["prompt"])

    if st.button("确认-需要重写"):
        st.session_state["platform"] = platform
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.session_state["prompt"] = prompt
        st.session_state["break_point"] = True
        st.rerun()
    if st.button("确认-不需要重写"):
        st.session_state["break_point"] = False
        st.rerun()


def graph_agent_page(api: ApiRequest, is_lite: bool = False):
    import rich

    # 初始化会话 id
    init_conversation_id()

    # 初始化模型配置
    if "platform" not in st.session_state:
        st.session_state["platform"] = "所有"
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = get_default_llm()
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.01
    if "prompt" not in st.session_state:
        st.session_state["prompt"] = ""
    if "article_generation_init_break_point" not in st.session_state:
        st.session_state["article_generation_init_break_point"] = False
    if "article_generation_repeat_break_point" not in st.session_state:
        st.session_state["article_generation_repeat_break_point"] = False

    with st.sidebar:
        tab1, = st.tabs(["工具设置"])
        with tab1:
            graph_names = list_graphs(api)
            selected_graph = st.selectbox(
                "选择工作流(必选)",
                graph_names,
                format_func=lambda x: x,
                key="selected_graph",
            )
            tools_list = list_tools(api)
            tool_names = ["None"] + list(tools_list)
            selected_tools = st.multiselect(
                "选择工具(可选)",
                list(tools_list),
                format_func=lambda x: tools_list[x]["title"],
                key="selected_tools",
            )
            selected_tool_configs = {
                name: tool["config"]
                for name, tool in tools_list.items()
                if name in selected_tools
            }

    selected_tools_configs = list(selected_tool_configs)

    st.title("自媒体文章生成")
    with st.chat_message("assistant"):
        st.write("Hello 👋, 我是自媒体文章生成 Agent, 试着向我提问.")

    with bottom():
        cols = st.columns([1, 0.2, 15, 1])
        if cols[0].button(":gear:", help="模型配置"):
            llm_model_setting()
        if cols[-1].button(":wastebasket:", help="清空对话"):
            st.session_state["messages"] = []
            st.rerun()
        user_input = cols[2].chat_input("请输入你的需求. 如: 请你帮我生成一篇自媒体文章.")

    # get_tool() 是所有工具的名称和对象的 dict 的列表
    all_tools = get_tool().values()
    tools = [tool for tool in all_tools if tool.name in selected_tools_configs]
    # 为保证调用效果, 如果用户没有选择任何 tool, 就默认选择互联网搜索工具
    if len(tools) == 0:
        search_internet = get_tool(name="search_internet")
        tools.append(search_internet)
    rich.print(tools)

    # 创建 llm 实例
    # todo: max_tokens 这里有问题, None 应该是不限制, 但是目前 llm 结果为 4096
    llm_model = st.session_state["llm_model"]
    llm = create_agent_models(configs=None,
                              model=llm_model,
                              max_tokens=None,
                              temperature=st.session_state["temperature"],
                              stream=True)
    rich.print(llm)

    if st.session_state.selected_graph == "文章生成":
        graph_name = "article_generation"
    else:
        graph_name = "base_graph"

    # 创建 langgraph 实例
    graph = get_graph_instance(
        name=graph_name,
        llm=llm,
        tools=tools,
        history_len=get_history_len(),
    )
    if not graph:
        raise ValueError(f"Graph '{graph_name}' is not registered.")

    # langgraph 配置文件
    graph_config = {
        "configurable": {
            "thread_id": st.session_state["conversation_id"]
        },
    }

    rich.print(graph_config)

    # 绘制流程图
    graph_png_image = graph.get_graph().draw_mermaid_png()
    st.sidebar.image(graph_png_image, caption="工作流流程图", use_column_width=True)

    # 创建 streamlit 消息缓存
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if graph_name == "article_generation":
        # 初始化文章和图片信息
        if "article_links" not in st.session_state:
            st.session_state["article_links"] = ""
        if "image_links" not in st.session_state:
            st.session_state["image_links"] = ""
        if "article_links_list" not in st.session_state:
            st.session_state["article_links_list"] = []
        if "image_links_list" not in st.session_state:
            st.session_state["image_links_list"] = []

    # 对话主流程
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Run the async function in a synchronous context
        graph_input = {"messages": [("user", user_input)]}
        asyncio.run(handle_user_input(graph_input=graph_input, graph=graph, graph_config=graph_config))

        # print("\n")
        # print(" ✅ ")
        # snapshot = graph.aget_state(graph_config)
        # print("This is snapshot:")
        # rich.print(snapshot)
        # print(" ✅ ✅ ")
        # print("\n")

    if graph_name == "article_generation":
        # 处理用户输入的链接
        article_links_list = [link.strip() for link in st.session_state["article_links"].split('\n') if link.strip()]
        image_links_list = [link.strip() for link in st.session_state["image_links"].split('\n') if link.strip()]
        st.session_state["article_links_list"] = article_links_list
        st.session_state["image_links_list"] = image_links_list
        article_links_list = st.session_state["article_links_list"]
        image_links_list = st.session_state["image_links_list"]
        logger.info(f"当前文章链接-处理后: {article_links_list}")
        logger.info(f"当前图片链接-处理后: {image_links_list}")

        is_article_generation_init_break_point = st.session_state["article_generation_init_break_point"]
        logger.info(f"是否断点: {str(is_article_generation_init_break_point)}")

        user_input = (f"article_links_list: {article_links_list}"
                      f"image_links_list: {image_links_list}")
        message_input = {
            "messages": HumanMessage(content=user_input),
            "user_feedback": user_input,
        }

        if st.session_state["article_generation_init_break_point"]:
            print("--State before update--")
            state = graph.aget_state(graph_config)
            rich.print(state)

            graph.aupdate_state(config=graph_config,
                                values=message_input,
                                as_node="article_generation_init_break_point")

            print("--State after update--")
            state = graph.aget_state(graph_config)
            rich.print(state)

            # asyncio.run(handle_user_input(graph_input=None, graph=graph, graph_config=graph_config))
