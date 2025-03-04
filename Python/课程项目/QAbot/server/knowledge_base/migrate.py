from configs import (EMBEDDING_MODEL, DEFAULT_VS_TYPE, ZH_TITLE_ENHANCE,
                     CHUNK_SIZE, OVERLAP_SIZE,
                    logger, log_verbose)
from server.knowledge_base.utils import (get_file_path, list_kbs_from_folder,
                                        list_files_from_folder,files2docs_in_thread,
                                        KnowledgeFile,)
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_file_repository import add_file_to_db
from server.db.base import Base, engine
import os
from typing import Literal, Any, List


def create_tables():
    Base.metadata.create_all(bind=engine)


def reset_tables():
    Base.metadata.drop_all(bind=engine)
    create_tables()


def file_to_kbfile(kb_name: str, files: List[str]) -> List[KnowledgeFile]:
    kb_files = []
    for file in files:
        try:
            kb_file = KnowledgeFile(filename=file, knowledge_base_name=kb_name)
            kb_files.append(kb_file)
        except Exception as e:
            msg = f"{e}，已跳过"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
    return kb_files


def folder2db(
    kb_names: List[str],
    mode: Literal["recreate_vs", "update_in_db", "increament"],
    vs_type: Literal["faiss", "milvus", "pg", "chromadb"] = DEFAULT_VS_TYPE,
    embed_model: str = EMBEDDING_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_SIZE,
    zh_title_enhance: bool = ZH_TITLE_ENHANCE,
):
    def files2vs(kb_name: str, kb_files: List[KnowledgeFile]):
        for success, result in files2docs_in_thread(kb_files,
                                                    chunk_size=chunk_size,
                                                    chunk_overlap=chunk_overlap,
                                                    zh_title_enhance=zh_title_enhance):
            if success:
                _, filename, docs = result
                print(f"正在将 {kb_name}/{filename} 添加到向量库，共包含{len(docs)}条文档")
                kb_file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
                kb_file.splited_docs = docs
                kb.add_doc(kb_file=kb_file, not_refresh_vs_cache=True)
            else:
                print(result)

    kb_names = kb_names or list_kbs_from_folder()
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service(kb_name, vs_type, embed_model)
        kb.create_kb()

        # 清除向量库，从本地文件重建
        if mode == "recreate_vs":
            kb.clear_vs()
            kb_files = file_to_kbfile(kb_name, list_files_from_folder(kb_name))
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        # 以数据库中文件列表为基准，利用本地文件更新向量库
        elif mode == "update_in_db":
            files = kb.list_files()
            kb_files = file_to_kbfile(kb_name, files)
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        # 对比本地目录与数据库中的文件列表，进行增量向量化
        elif mode == "increament":
            db_files = kb.list_files()
            folder_files = list_files_from_folder(kb_name)
            files = list(set(folder_files) - set(db_files))
            kb_files = file_to_kbfile(kb_name, files)
            files2vs(kb_name, kb_files)
            kb.save_vector_store()
        else:
            print(f"unspported migrate mode: {mode}")


def prune_db_docs(kb_names: List[str]):
    '''
    delete docs in database that not existed in local folder.
    it is used to delete database docs after user deleted some doc files in file browser
    '''
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb and kb.exists():
            files_in_db = kb.list_files()
            files_in_folder = list_files_from_folder(kb_name)
            files = list(set(files_in_db) - set(files_in_folder))
            kb_files = file_to_kbfile(kb_name, files)
            for kb_file in kb_files:
                kb.delete_doc(kb_file, not_refresh_vs_cache=True)
                print(f"success to delete docs for file: {kb_name}/{kb_file.filename}")
            kb.save_vector_store()


def prune_folder_files(kb_names: List[str]):
    '''
    delete doc files in local folder that not existed in database.
    is is used to free local disk space by delete unused doc files.
    '''
    for kb_name in kb_names:
        kb = KBServiceFactory.get_service_by_name(kb_name)
        if kb and kb.exists():
            files_in_db = kb.list_files()
            files_in_folder = list_files_from_folder(kb_name)
            files = list(set(files_in_folder) - set(files_in_db))
            for file in files:
                os.remove(get_file_path(kb_name, file))
                print(f"success to delete file: {kb_name}/{file}")
