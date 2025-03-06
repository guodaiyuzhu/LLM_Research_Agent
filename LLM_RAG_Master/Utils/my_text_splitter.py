import re
from langchain.text_splitter import CharacterTextSplitter
from typing import List

class MyTextSplitter(CharacterTextSplitter):
    def __init__(self, chunk_size: int = 150, chunk_overlap: int = 20, **kwargs):
        super().__init__(**kwargs)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        splits = []
        texts = re.split(r'\n{1,}', text)
        for text in texts:
            start_idx = 0
            end_overlap_idx = start_idx + self._chunk_size + self._chunk_overlap
            cur_idx = min(start_idx + self._chunk_size, len(text))
            chunk_text = text[start_idx:cur_idx]
            splits.append(chunk_text)
            while end_overlap_idx < len(text):
                start_idx += self._chunk_overlap
                cur_idx = min(start_idx + self._chunk_size, len(text))
                end_overlap_idx = start_idx + self._chunk_size + self._chunk_overlap
                chunk_text = text[start_idx:cur_idx]
                splits.append(chunk_text)
        return splits

# filepath = r'../knowledge/TXT/'

# from tqdm import tqdm
# from utils.Tools import tree
# from langchain.document_loaders import TextLoader

# for fullfilepath, file in tqdm(zip(*tree(filepath)), desc="加载文件"):
#     loader = TextLoader(fullfilepath, autodetect_encoding=True)
#     textsplitter = MyTextSplitter()
#     docs = loader.load_and_split(textsplitter)
