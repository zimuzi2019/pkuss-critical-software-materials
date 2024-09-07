import os

from transformers import AutoTokenizer

from configs import (
    EMBEDDING_MODEL,
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    text_splitter_dict,
    LLM_MODEL,
    TEXT_SPLITTER_NAME,
)
import importlib
from text_splitter import zh_title_enhance as func_zh_title_enhance
import langchain.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from server.utils import run_in_thread_pool, embedding_device, get_model_worker_config
import io
from typing import List, Union, Callable, Dict, Optional, Tuple, Generator
import chardet


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), vector_name)


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def list_kbs_from_folder():
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    return [file for file in os.listdir(doc_path)
            if os.path.isfile(os.path.join(doc_path, file))]


def load_embeddings(model: str = EMBEDDING_MODEL, device: str = embedding_device()):
    '''
    从缓存中加载embeddings，可以避免多线程时竞争加载。
    '''
    from server.knowledge_base.kb_cache.base import embeddings_pool
    return embeddings_pool.load_embeddings(model=model, device=device)


LOADER_DICT = {
               "RapidOCRPDFLoader": [".pdf"],
               "UnstructuredFileLoader": ['.eml', '.msg', '.rst',
                                          '.rtf', '.txt', '.xml',
                                          '.docx', '.epub', '.odt',
                                          '.ppt', '.pptx', '.tsv'],
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass



def get_loader(loader_name: str, file_path_or_content: Union[str, bytes, io.StringIO, io.BytesIO]):
    try:
        if loader_name == "RapidOCRPDFLoader":
            document_loaders_module = importlib.import_module('document_loaders')
        else:
            document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"为文件{file_path_or_content}查找加载器{loader_name}时出错：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader = DocumentLoader(file_path_or_content, autodetect_encoding=True)
    elif loader_name == "UnstructuredHTMLLoader":
        loader = DocumentLoader(file_path_or_content, mode="elements")
    else:
        loader = DocumentLoader(file_path_or_content)
    return loader


def make_text_splitter(
    splitter_name: str = TEXT_SPLITTER_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = OVERLAP_SIZE,
    llm_model: str = LLM_MODEL,
):
    splitter_name = splitter_name or "SpacyTextSplitter"

    try:  # 自定义text_splitter
        text_splitter_module = importlib.import_module('text_splitter')
        # print("="*30+'fail2'+'='*30)
        TextSplitter = getattr(text_splitter_module, splitter_name)
        # print("="*30+'fail'+'='*30)
    except:  ## 调用失败则使用langchain的text_splitter
        # print(splitter_name)
        print("="*30+'custom fail'+'='*30)
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, splitter_name)
    if text_splitter_dict[splitter_name]["source"] == "tiktoken":  ## 从tiktoken加载
        # print("="*30+'in tiktoken'+'='*30)
        try:
            text_splitter = TextSplitter.from_tiktoken_encoder(
                encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                pipeline="zh_core_web_sm",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except:
            # print("="*30+'pipeline fail'+'='*30)
            text_splitter = TextSplitter.from_tiktoken_encoder(
                encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
    return text_splitter

class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_base_name: str
    ):
        self.kb_name = knowledge_base_name
        self.filename = filename
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.ext}")
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.splited_docs = None
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = TEXT_SPLITTER_NAME

    def file2docs(self, refresh: bool=False):
        if self.docs is None or refresh:
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            loader = get_loader(self.document_loader_name, self.filepath)
            self.docs = loader.load()
        return self.docs

    def docs2texts(
        self,
        docs: List[Document] = None,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(splitter_name=self.text_splitter_name, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            docs = text_splitter.split_documents(docs)

        print(f"文档切分示例：{docs[0]}")
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs

    def file2text(
        self,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(docs=docs,
                                                zh_title_enhance=zh_title_enhance,
                                                refresh=refresh,
                                                chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                text_splitter=text_splitter)
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
        pool: ThreadPoolExecutor = None,
) -> Generator:
    def file2docs(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return False, (file.kb_name, file.filename, msg)

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename=file[0]
                kb_name=file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in run_in_thread_pool(func=file2docs, params=kwargs_list, pool=pool):
        yield result


if __name__ == "__main__":
    from pprint import pprint

    kb_file = KnowledgeFile(filename="test.txt", knowledge_base_name="samples")
    docs = kb_file.file2docs()
    pprint(docs[-1])

    docs = kb_file.file2text()
    pprint(docs[-1])
