import uuid

import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
    get_tool,
)

logger = build_logger()


def init_conversation_id():
    # 检查是否已经在 session_state 中存储了 UUID
    if "conversation_id" not in st.session_state:
        # 生成一个随机的UUID并存储在 session_state 中
        st.session_state["conversation_id"] = str(uuid.uuid4())


@st.experimental_dialog("模型配置", width="large")
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
    temperature = cols[2].slider("温度设置(Temperature)", 0.0, 1.0, value=0.01)
    system_message = st.text_area("指令(Prompt):")

    if st.button("OK"):
        # 保存状态到 session_state
        st.session_state["platform"] = platform
        st.session_state["llm_model"] = llm_model
        st.session_state["temperature"] = temperature
        st.session_state["system_message"] = system_message
        st.rerun()


def graph_agent_page(
    api: ApiRequest,
    is_lite: bool = False,
):
    import rich  # debug

    # 初始化 conversation_id
    init_conversation_id()

    # 初始化 session_state 中的键
    if "platform" not in st.session_state:
        st.session_state["platform"] = "所有"
    if "llm_model" not in st.session_state:
        st.session_state["llm_model"] = get_default_llm()
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.01
    if "system_message" not in st.session_state:
        st.session_state["system_message"] = ""

    with st.sidebar:
        tab1, = st.tabs(["工具设置"])

        with tab1:
            # 选择 langgraph 模板
            graph_names = list_graphs(api) + ["None"]
            selected_graph = st.selectbox(
                "选择工作流(必选)",
                graph_names,
                format_func=lambda x: "None" if x == "None" else x,
                key="selected_graph",
            )

            # 选择工具
            tools = list_tools(api)
            tool_names = ["None"] + list(tools)
            selected_tools = st.multiselect(
                "选择工具(可选)",
                list(tools),
                format_func=lambda x: tools[x]["title"],
                key="selected_tools",
            )
            selected_tool_configs = {
                name: tool["config"]
                for name, tool in tools.items()
                if name in selected_tools
            }

    # 选择的工具
    selected_tools_configs = list(selected_tool_configs)

    st.title("自媒体文章生成")
    with st.chat_message("assistant"):
        st.write("Hello 👋, 我是自媒体文章生成 Agent, 试着向我提问.")

    # chat input
    with bottom():
        cols = st.columns([1, 0.2, 15,  1])
        if cols[0].button(":gear:", help="模型配置"):
            llm_model_setting()
        if cols[-1].button(":wastebasket:", help="清空对话"):
            st.session_state["message"] = []
            st.rerun()
        # Accept user input
        user_input = cols[2].chat_input("请输入你的需求. 如: 请你帮我生成一篇自媒体文章.")

    # debug
    with st.status("debug info", expanded=True):
        st.write("当前 llm 平台:", st.session_state["platform"])
        st.write("当前 llm 模型:", st.session_state["llm_model"])
        st.write("当前 llm 温度:", st.session_state["temperature"])
        st.write("当前系统 prompt:", st.session_state["system_message"])
        st.write("当前 tools:", selected_tools_configs)
        st.write("当前会话的 id:", st.session_state["conversation_id"])
        if st.session_state.selected_graph == "文章生成":
            st.write("当前工作流:", "article_generation")
        elif st.session_state.selected_graph == "通用机器人":
            st.write("当前工作流:", "base_graph")

    all_tools = get_tool().values()
    tools = [tool for tool in all_tools if tool.name in selected_tools_configs]
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

    graph = get_graph_instance(
        name=graph_name,
        llm=llm,
        tools=tools,
        history_len=get_history_len(),
    )
    if not graph:
        raise ValueError(f"Graph '{graph_name}' is not registered.")

    graph_config = {
        "configurable": {
            "thread_id": st.session_state["conversation_id"]
        },
        "recursion_limit": get_recursion_limit()
    }

    rich.print(graph)

    with st.status("debug graph info", expanded=True):
        st.write("当前 graph:", graph)
        st.write("当前 graph_config:", graph_config)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)
        # Add user message to chat history
        st.session_state.messages.append(HumanMessage(content=user_input))

        # 对接 langgraph
        events = graph.stream(
            {"messages": [("user", user_input)]}, graph_config, stream_mode="updates"
        )
        if events:
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                for event in events:
                    st.markdown(event["messages"])
                    # st.json(response, expanded=True)
                # Add assistant response to chat history
                st.session_state.messages.append(AIMessage(content=event["messages"]))
