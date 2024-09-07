import sys
from configs.model_config import LLM_DEVICE


HTTPX_DEFAULT_TIMEOUT = 300.0

OPEN_CROSS_DOMAIN = False

DEFAULT_BIND_HOST = "localhost"

WEBUI_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 14001,
}

API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 14000,
}

FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 20000,
}

FSCHAT_MODEL_WORKERS = {
    # 所有模型共用的默认配置，可在模型专项配置中进行覆盖。
    "default": {
        "host": DEFAULT_BIND_HOST,
        "port": 20002,
        "device": LLM_DEVICE,
        "infer_turbo": "vllm" if sys.platform.startswith("linux") else False,

    },
    "zhipu-api": { # 请为每个要运行的在线API设置不同的端口
        "port": 21001,
    },
}

FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 20001,
    "dispatch_method": "shortest_queue",
}