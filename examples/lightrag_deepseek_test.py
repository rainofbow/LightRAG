import os
import asyncio
import inspect
import logging
import logging.config
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./dickens"

# 配置日志
def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 使用兼容OpenAI的API定义LLM模型函数
async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        # api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        api_key="sk-18c7f8583dc846e7b5a155fa7915c04b",
        # base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
        base_url="https://api.deepseek.com",
        **kwargs,
    )

# 异步打印流式响应
async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)

# 初始化RAG实例
async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            # embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            embedding_dim=1024,
            # max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts,
                # embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                embed_model="bge-m3:latest",
                # host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
                host="http://localhost:11434",
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def main():
    rag = None
    try:
        # Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]
        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        with open("./book.txt", "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        print("\n=====================")
        print("进入循环提问模式，直接输入问题，回车发送，输入为空退出。\n")
        print("可选模式: naive, local, global, hybrid。默认naive。输入格式: [模式:]你的问题\n例如: local:请总结故事内容\n")
        while True:
            user_input = input("请输入你的问题（或回车退出）：").strip()
            if not user_input:
                print("退出循环。"); break
            # 支持模式选择
            if ":" in user_input:
                mode, query = user_input.split(":", 1)
                mode = mode.strip().lower()
                query = query.strip()
                if mode not in ["naive", "local", "global", "hybrid"]:
                    print("无效模式，使用默认naive。")
                    mode = "naive"
            else:
                mode = "naive"
                query = user_input
            print(f"\nQuery mode: {mode}")
            print(f"Question: {query}")
            try:
                resp = await rag.aquery(query, param=QueryParam(mode=mode, stream=True))
                if inspect.isasyncgen(resp):
                    await print_stream(resp)
                else:
                    print(resp)
                print("\n----------------------\n")
            except Exception as e:
                print(f"查询出错: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")

