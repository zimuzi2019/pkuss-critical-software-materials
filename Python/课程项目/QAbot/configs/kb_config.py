import os


# 默认向量库类型为FAISS
DEFAULT_VS_TYPE = "faiss"

# 缓存向量库数量
CACHED_VS_NUM = 1

# 知识库中单段文本长度(不适用MarkdownHeaderTextSplitter)
CHUNK_SIZE = 250

# 知识库中相邻文本重合长度(不适用MarkdownHeaderTextSplitter)
OVERLAP_SIZE = 50

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 3

# 知识库匹配相关度阈值
SCORE_THRESHOLD = 0.6

ZH_TITLE_ENHANCE = False

# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
if not os.path.exists(KB_ROOT_PATH):
    os.mkdir(KB_ROOT_PATH)

# 数据库默认存储路径。
DB_ROOT_PATH = os.path.join(KB_ROOT_PATH, "info.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"

# 可选向量库类型及对应配置
kbs_config = {
    "faiss": {
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "secure": False,
    },
    "pg": {
        "connection_uri": "postgresql://postgres:postgres@127.0.0.1:5432/langchain_chatchat",
    }
}

# TextSplitter配置
text_splitter_dict = {
    "ChineseRecursiveTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "gpt2",
    },
}

TEXT_SPLITTER_NAME = "ChineseRecursiveTextSplitter"
