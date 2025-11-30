import json
import numpy as np
import faiss
from collections import defaultdict

TREE_FILE = 'vector_tree_db.json'
TREE_WITH_INDEX_FILE = 'index_tree_db.json'
FAISS_INDEX_FILE = 'faiss.index'

def extract_and_index(node, all_vectors, idx_counter):
    """재귀적으로 벡터 수집 + vector_idx 추가"""
    if isinstance(node, dict):
        for k, v in node.items():
            extract_and_index(v, all_vectors, idx_counter)
    elif isinstance(node, list):
        for item in node:
            if 'vector' in item:

                item['vector_idx'] = idx_counter[0]  # tree_db_with_index용 index
                all_vectors.append(item['vector'])
                idx_counter[0] += 1
                del item['vector']  # 딕셔너리에서 벡터 제거

def build_faiss_index():
    with open(TREE_FILE, 'r', encoding='utf-8') as f:
        db_tree = json.load(f)

    idx_counter = [0]
    all_vectors = []
    extract_and_index(db_tree, all_vectors, idx_counter)

    # FAISS 인덱스 생성
    vectors_np = np.array(all_vectors).astype('float32')
    faiss.normalize_L2(vectors_np)
    dim = vectors_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors_np)

    faiss.write_index(index, FAISS_INDEX_FILE)
    print(f"✅ FAISS 인덱스 저장 완료 → {FAISS_INDEX_FILE}, 벡터 수: {len(all_vectors)}")

    # tree_db_with_index 저장
    with open(TREE_WITH_INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(db_tree, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON에 vector_idx 추가 → {TREE_WITH_INDEX_FILE}")

if __name__ == "__main__":
    build_faiss_index()
