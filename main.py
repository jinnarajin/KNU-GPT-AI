'''클라이언트 → 서버 (입력 JSON)
{
    "Private_Info": { 
        "Name": "홍길동",
        "Student_ID": 2000000000
    },
    "Path": ["IT대학", "전자공학부", "장학금"],
    "Question": "성적과 상관없이 받을 수 있는 장학금이 뭐가 있어?"
}
'''

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import uvicorn
import json
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from fastapi.responses import StreamingResponse
import re

# ================== 서버 기본 설정 ==================
app = FastAPI()

# 개발용으로 모든 도메인 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_PATH = "faiss.index" 
TREE_DB_PATH = "index_tree_db.json"
CONFIG_PATH = "config.json"

# ================== API 키 불러오기 ==================
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
        GEMINI_API_KEY = config.get("GEMINI_API_KEY")
except Exception as e:
    raise RuntimeError("❌ config.json 파일을 읽을 수 없습니다.") from e

if not GEMINI_API_KEY:
    raise ValueError("❌ Gemini API Key가 config.json에 설정되지 않았습니다.")

# ✅ Gemini 설정
genai.configure(api_key=GEMINI_API_KEY)

# ================== 모델 & 인덱스 로드 ==================
print("🚀 모델 로드 중...")
model = SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")

print("🚀 FAISS 인덱스 로드 중...")
index = faiss.read_index(INDEX_PATH)

with open(TREE_DB_PATH, "r", encoding="utf-8") as f:
    TREE_DB = json.load(f)

print("✅ 검색 서버 준비 완료!")

# ================== 레코드 수집 (경로 포함) ==================
def collect_records_with_path(node, path_prefix=None):
    records = []
    if path_prefix is None:
        path_prefix = []

    if isinstance(node, dict):
        for key, value in node.items():
            records.extend(collect_records_with_path(value, path_prefix + [key]))
    elif isinstance(node, list):
        for r in node:
            record = r.copy()
            record["path"] = path_prefix
            records.append(record)
    return records

# ================== 안전한 경로 탐색 ==================
def get_node_by_path(tree, path):
    node = tree
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


# ================== 검색 함수 ==================
def semantic_search(path, question, top_k, threshold=0.5):
    node = get_node_by_path(TREE_DB, path)

    # 지정된 path가 없으면 전체 탐색
    if node is None:
        records = collect_records_with_path(TREE_DB)
    else:
        # 해당 노드 및 모든 하위 노드 수집
        records = collect_records_with_path(node, path)

    if not records:
        return []

    idxs = [r["vector_idx"] for r in records]
    vectors = np.array([index.reconstruct(i) for i in idxs]).astype("float32")
    faiss.normalize_L2(vectors)

    q_vec = model.encode(question, normalize_embeddings=True).astype("float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)

    sub_index = faiss.IndexFlatIP(vectors.shape[1])
    sub_index.add(vectors)
    D, I = sub_index.search(q_vec, len(records))

    results = [
        {
            "score": float(D[0][i]),
            "title": records[I[0][i]]["title"],
            "paragraph": records[I[0][i]]["paragraph"],
            "path": records[I[0][i]]["path"],
        }
        for i in range(len(records)) if D[0][i] >= threshold
    ]

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]

# ================== /search 엔드포인트 ==================
@app.post("/chat")
async def search(request: Request):
    try:
        data = await request.json()

        question = data.get("question", "").strip()
        previous_chat_histories = data.get("previous_chat_histories", [])
        path = data.get("path", [])
        top_k = data.get("top_k", 5)
        threshold = data.get("threshold", 0.5)
        private_info = data.get("private_Info", {})

        if not question:
            return JSONResponse({"error": "Question 이 누락됨"}, status_code=400)

        search_results = semantic_search(path, question, top_k, threshold)

        db_text = json.dumps(search_results, ensure_ascii=False, indent=4)

        now = datetime.now()

        prompt = f"""
너는 경북대학교에 대한 정보를 안내 해주는 챗봇임
다음은 사용자의 개인 정보와 이전 대화 기록, 질문이며 비어있을 수 있으니 있는 정보를 최대한 이용해아하며
주어진 정보를 참고해서 상대에게 친절하고 간결하게 한국어로 답변을 작성해야함.
검색된 데이터베이스가 있다면 해당 정보를 자세히 안내
검색한 데이터베이스가 없다면 

현재 날짜, 시간
{now}

사용자 개인 정보:
{private_info}

이전 대화 기록:
{previous_chat_histories}

📌 학생 질문:
"{question}"

📚 검색된 데이터베이스 (참고용):
{db_text}

✏️ 작성 지침:
- 학생에게 직접 설명하듯이 말해줘.
- 마크다운 형식으로 대답해줘.
- URL 정보가 있다면 이를 [url정보](실제 URL) 형식으로 나타내줘.
- 검색된 데이터베이스가 없다면 정확하지 않을 수 있다고 안내해줘.
                """

        # ================== LLM 호출 ==================
        model_gemini = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash"
        )
        response = model_gemini.generate_content(prompt)
        answer_text = response.text.strip() if response.text else "응답을 생성하지 못했습니다."

        return {
            "answer": answer_text
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

        #
        # def stream_geminai_answer():
        #     try:
        #         response = model_gemini.generate_content(
        #             contents=prompt,
        #             stream=True,
        #             generation_config={
        #                 "temperature": 0.5,
        #                 "max_output_tokens": 2000
        #             },
        #         )
        #
        #         for chunk in response:
        #             text = None
        #
        #             # ▣ Case 1: chunk.text directly exists
        #             if hasattr(chunk, "text") and chunk.text:
        #                 text = chunk.text
        #
        #             # ▣ Case 2: parts[] 안에서 텍스트 추출
        #             if not text and hasattr(chunk, "candidates"):
        #                 for c in chunk.candidates:
        #                     content = getattr(c, "content", None)
        #                     if content:
        #                         for p in getattr(content, "parts", []):
        #                             if hasattr(p, "text") and p.text:
        #                                 text = p.text
        #                                 break
        #
        #             # text가 존재하면 SSE-safe 변환 후 전송
        #             if text:
        #                 # ⭐ 줄바꿈을 먼저 "\\n" 문자로 치환해서 '공백이 아닌 문자'로 만든다
        #                 marked = text.replace("\n", "\\n")
        #
        #                 # 문장 단위로 자르기 (이제 \n은 더 이상 공백이 아니라서 안 날아감)
        #                 sentences = [
        #                     s.strip()
        #                     for s in re.split(r'(?<=[.!?])\s+', marked)
        #                     if s.strip()
        #                 ]
        #
        #                 for sentence in sentences:
        #                     # 단어 단위로 자르기
        #                     words = sentence.split(" ")
        #
        #                     for w in words:
        #                         # 여기서는 w 안에 "\\n" 이 그대로 들어 있음
        #                         safe_text = json.dumps(w, ensure_ascii=False)[1:-1]
        #                         yield f"data: {safe_text}\n\n"
        #
        #         yield "data: [END]\n\n"
        #
        #     except Exception as e:
        #         yield f"data: [ERROR] {str(e)}\n\n"
        #
        #
        # # ✅ 클라이언트로 바로 텍스트 스트리밍 반환
        # return StreamingResponse(stream_geminai_answer(), media_type="text/event-stream")




@app.post("/title")
async def title(request: Request):
    try:

        data = await request.json()
        query = data.get("question", "")
        prompt = f"""
        다음은 학생이 한 질문 이며 이를 10자 이내로 요약 하여 채팅방의 제목으로 사용하려고 합니다..        

        📚 학생이 한 질문:
        {query}

        ✏️ 작성 지침:
        - 질문의 핵심을 요약해서 10자 이내로 만들어줘.
                        """

        # ================== LLM 호출 ==================
        model_gemini = genai.GenerativeModel(
            model_name="models/gemini-2.5-flash"
        )
        response = model_gemini.generate_content(prompt)
        answer_text = response.text.strip() if response.text else None

        return {
            "answer": answer_text
        }


    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



# ================== 서버 실행 ==================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
