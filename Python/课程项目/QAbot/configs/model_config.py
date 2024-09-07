import os

MODEL_ROOT_PATH=''

MODEL_PATH = {
    "embed_model": {
        "m3e-base": r"m3e-base",
    },

    "llm_model": {
        "chatglm2-6b": r"chatglm2-6b",
    },
}


EMBEDDING_MODEL = "m3e-base"
EMBEDDING_DEVICE = "cuda"

# LLM_MODEL = "chatglm2-6b"
LLM_MODEL = "zhipu-api"
LLM_DEVICE = "auto"

# 历史对话轮数
HISTORY_LEN = 3

TEMPERATURE = 0.7


ONLINE_LLM_MODEL = {
    "zhipu-api": {
        "api_key": "b8d65675468364ee92988ca0e44af2f5.waoJaduliKBkETuP",
        "version": "chatglm_pro",  # 可选 "chatglm_lite", "chatglm_std", "chatglm_pro"
        "provider": "ChatGLMWorker",
    },
}

# nltk 模型存储路径
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")