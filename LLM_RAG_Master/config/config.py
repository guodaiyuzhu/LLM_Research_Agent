import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EMBEDDING_MODEL = "/home/jerryxu/Downloads/LLM/LLM_ChatIntent_Pro/m3e-base"
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MILVUS_CONNECTION = {
    "host": "192.168.2.105",# 192.168.2.102 # 192.168.2.105
    "port": "19530",
    "user": "",
    "password": "",
    "secure": False
}

DEFAULT_COLLECTION_NAME = 'FICC_WIKI_TEST_5_0'
DEFAULT_INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "IVF_PQ",
    "params": {
        "nlist": 512,
        "m": 4,
        "nbits": 8
    }
}

DEFAULT_SEARCH_PARAMS = {
    "metric_type": "L2",
    "params": {"nlist": 10}
}

# Chunking parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Whether to enable Chinese title enhancement
ZH_TITLE_ENHANCE = False

# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_DEFAULT_MODEL = 'qwen2:7b'

# GreenPlum connection information
DEFAULT_GP_CONNECTION = {
    "host": "168.63.25.66",
    "port": "5432",
    "database": "headdw",
    "user": "ficcdw_gp",
    "password": "RmLljY18ybzlyX1VzZXI="
}

# Oracle connection information
DEFAULT_ORACLE_CONNECTION = {
    "host": "168.63.18.146",
    "port": "1521",
    "database": "heads",
    "user": "ficc_dwods",
    "password": "TEhBS19scXBnXzMyNjQ="
}
