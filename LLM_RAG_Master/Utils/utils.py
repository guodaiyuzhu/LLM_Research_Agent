from LLM_RAG_Master.config.config import *
from sentence_transformers import SentenceTransformer, util

def tree(filepath, ignore_dir_names=None, ignore_file_names=None):
    """
    遍历同等分支，第一个为文件路径，第二个为对应的文件名
    """
    if ignore_dir_names is None:
        ignore_dir_names = []
    if ignore_file_names is None:
        ignore_file_names = []
    ret_list = []
    if isinstance(filepath, str):
        if not os.path.exists(filepath):
            print("路径不存在")
            return None
        elif os.path.isfile(filepath) and os.path.basename(filepath) not in ignore_file_names:
            return [filepath]
        elif os.path.isdir(filepath) and os.path.basename(filepath) not in ignore_dir_names:
            for file in os.listdir(filepath):
                fullfilepath = os.path.join(filepath, file)
                if (os.path.isfile(fullfilepath) and os.path.basename(fullfilepath) not in ignore_file_names):
                    ret_list.append(fullfilepath)
                if (os.path.isdir(fullfilepath) and os.path.basename(fullfilepath) not in ignore_dir_names):
                    ret_list.extend(tree(fullfilepath, ignore_dir_names, ignore_file_names))
    return ret_list

def sim_score(embeddings_client: SentenceTransformer, query: str, search_result: list):
    """计算两个句子的相似度分数"""
    query_vec = embeddings_client.encode(query)
    # 查询向量嵌入
    search_vec_list = embeddings_client.encode(search_result)
    # 通过cosine计算相似度
    socre_list = ["{0:.4f}".format(util.cos_sim(query_vec, search_vec_list[i]).tolist()[0][0]) for i in range(len(search_vec_list))]
    # 结果和得分组合形成列表
    result = [(search_result[i], socre_list[i]) for i in range(len(socre_list))]
    return result

def sort_list(lst):
    """对列表进行U型排序"""
    sorted_lst = sorted(lst, key=lambda i: i[1], reverse=True)
    left_list = []
    right_list = []
    k: int = len(sorted_lst)
    half_k = k // 2
    for j in range(1, k + 1, 1):
        if j <= half_k:
            right_list.append(sorted_lst[-j])
        else:
            left_list.append(sorted_lst[-j])
    left_list = sorted(left_list, key=lambda i: i[1], reverse=True)
    left_list.extend(right_list)
    return left_list

def torch_gc():
    if torch.cuda.is_available():
        # 清除CUDA缓存和IPC
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(e)
            print("如果您使用的是 macOS 建议将 pytorch 版本升级至 2.0.0 或更高版本，以支持及时清理 torch 产生的内存占用。")
