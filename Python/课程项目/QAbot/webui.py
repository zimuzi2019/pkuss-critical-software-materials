import streamlit as st
from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages import *
import os
from configs import VERSION
from server.utils import api_address


api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    st.set_page_config(
        "Langchain-Chatchat WebUI",
        os.path.join("img", "icon.png"),
        initial_sidebar_state="expanded",
        menu_items={
            'About': f"""欢迎使用 QAbot v0.0.1！"""
        }
    )

    if not chat_box.chat_inited:
        st.toast(
            f"当前使用模型`{LLM_MODEL}`, 您可以开始提问了."
        )

    pages = {
        "对话": {
            "icon": "chat",
            "func": dialogue_page,
        },
        "知识库管理": {
            "icon": "hdd-stack",
            "func": knowledge_base_page,
        },
    }

    with st.sidebar:
        st.image(
            os.path.join(
                "img",
                "logo.png"
            ),
            use_column_width=True
        )
        st.caption(
            f"""<p align="right">v0.0.1</p>""",
            unsafe_allow_html=True,
        )
        options = list(pages)
        icons = [x["icon"] for x in pages.values()]

        default_index = 0
        selected_page = option_menu(
            "",
            options=options,
            icons=icons,
            # menu_icon="chat-quote",
            default_index=default_index,
        )

    if selected_page in pages:
        pages[selected_page]["func"](api)
