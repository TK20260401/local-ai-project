# ====================================================================
# Lumina RAG (local OCR + hybrid search) — 検索網拡大 & 矛盾併記版
# ====================================================================

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import chromadb
import docx
import easyocr
import numpy as np
import ollama
import pdf2image
import streamlit as st
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ==== 環境に合わせて調整する定数 ====
DATA_DIR = "../data"
COLLECTION_NAME = "lumina_manual_v15"
LLM_MODEL = "qwen3.5-9b-16k:latest"
EMBED_MODEL = "bge-m3"
# ==================================


@st.cache_resource
def load_reranker():
    print(">>> load_reranker start", flush=True)
    r = CrossEncoder("BAAI/bge-reranker-base")
    print(">>> load_reranker done", flush=True)
    return r


@st.cache_resource
def load_ocr_reader():
    print(">>> load_ocr_reader start", flush=True)
    r = easyocr.Reader(["ja", "en"], gpu=False)
    print(">>> load_ocr_reader done", flush=True)
    return r


# PDF抽出（Force OCR：全ページ強制OCR）
def extract_text_from_pdf(pdf_path):
    try:
        print(f">>>   OCR converting: {os.path.basename(pdf_path)}", flush=True)
        ocr_reader = load_ocr_reader()
        images = pdf2image.convert_from_path(pdf_path)
        print(f">>>   {len(images)} pages", flush=True)
        text = ""
        for i, img in enumerate(images):
            print(f">>>   OCR page {i + 1}/{len(images)}", flush=True)
            img_np = np.array(img)
            result = ocr_reader.readtext(img_np, detail=0)
            text += "\n".join(result) + "\n"
        return text
    except Exception as e:
        print(f">>>   OCR ERROR: {e}", flush=True)
        return ""


def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception as e:
        print(f">>>   DOCX ERROR: {e}", flush=True)
        return ""


def recursive_split_japanese(text, max_chars=300, overlap=50):
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    chunks = []
    current_chunk = ""
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(current_chunk) + len(paragraph) <= max_chars:
            current_chunk += ("\n" + paragraph) if current_chunk else paragraph
        else:
            for sentence in paragraph.split("。"):
                sentence = sentence.strip()
                if not sentence:
                    continue
                sentence_with_period = sentence + "。"
                if len(current_chunk) + len(sentence_with_period) <= max_chars:
                    current_chunk += sentence_with_period
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    overlap_text = (
                        current_chunk[-overlap:]
                        if len(current_chunk) > overlap
                        else current_chunk
                    )
                    current_chunk = overlap_text + sentence_with_period
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def embed_texts(texts):
    resp = ollama.embed(model=EMBED_MODEL, input=texts)
    return resp["embeddings"]


@st.cache_resource
def prepare_knowledge():
    print(">>> prepare_knowledge START", flush=True)
    chroma_client = chromadb.PersistentClient(path="./.chromadb_data")
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() > 0:
        print(">>> existing collection found, loading", flush=True)
        stored = collection.get()
        print(">>> prepare_knowledge END (from cache)", flush=True)
        return chroma_client, collection, stored["documents"], stored["metadatas"]

    all_docs, all_metadatas = [], []

    if os.path.exists(DATA_DIR):
        for filename in os.listdir(DATA_DIR):
            print(f">>> processing file: {filename}", flush=True)
            file_path = os.path.join(DATA_DIR, filename)
            file_text = ""
            if filename.endswith(".pdf"):
                file_text = extract_text_from_pdf(file_path)
            elif filename.endswith(".docx"):
                file_text = extract_text_from_docx(file_path)
            elif filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    file_text = f.read()
            else:
                print(f">>>   skip (unsupported): {filename}", flush=True)

            if file_text.strip():
                for chunk in recursive_split_japanese(file_text, 300, 50):
                    all_docs.append(chunk)
                    all_metadatas.append({"file_name": filename})
            print(f">>> done file: {filename}", flush=True)
    else:
        print(f">>> DATA_DIR not found: {DATA_DIR}", flush=True)
        st.warning(f"⚠️ データフォルダが見つかりません: {DATA_DIR}")

    if all_docs:
        print(f">>> embedding {len(all_docs)} chunks ...", flush=True)
        ids = [f"doc_{i}" for i in range(len(all_docs))]
        embeddings = embed_texts(all_docs)
        print(">>> embedding done, writing to chroma ...", flush=True)
        collection.add(
            documents=all_docs,
            ids=ids,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )
        print(">>> chroma write done", flush=True)

    print(">>> prepare_knowledge END", flush=True)
    return chroma_client, collection, all_docs, all_metadatas


def search(prompt, target_docs, collection, selected_file, reranker):
    if not target_docs:
        return [], "指定されたファイルにデータがありません。"

    # 候補を広めに集める（複合質問で片方が溢れないよう網を拡大：15→30）
    bm25 = BM25Okapi([list(doc) for doc in target_docs])
    bm25_scores = bm25.get_scores(list(prompt))
    top_k = min(30, len(target_docs))
    bm25_idx = sorted(
        range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
    )[:top_k]
    candidates = [target_docs[i] for i in bm25_idx]

    try:
        q_emb = embed_texts([prompt])[0]
        where = (
            None
            if selected_file == "すべてのファイル"
            else {"file_name": selected_file}
        )
        vres = collection.query(
            query_embeddings=[q_emb],
            n_results=min(30, len(target_docs)),
            where=where,
        )
        candidates += vres["documents"][0]
    except Exception as e:
        print(f">>> vector search skipped: {e}", flush=True)

    candidates = list(dict.fromkeys(candidates))

    # リランクで関連度順に並べ替え、上位6件を採用（5→6）
    pairs = [[prompt, doc] for doc in candidates]
    rerank_scores = reranker.predict(pairs)
    scored = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)
    final_docs = scored[:6]
    context = "\n---\n".join(doc for _, doc in final_docs)
    return final_docs, context


# ==== 実行本体 ====

print(">>> APP START", flush=True)
reranker = load_reranker()
chroma_client, collection, all_docs, all_metadatas = prepare_knowledge()
print(">>> knowledge ready, drawing UI", flush=True)

st.title("🎯 Lumina RAGチャット (ローカルOCR＆再帰分割 搭載版)")

st.sidebar.header("📁 検索対象のフィルタリング")
unique_files = (
    sorted(set(m["file_name"] for m in all_metadatas)) if all_metadatas else []
)
file_options = ["すべてのファイル"] + unique_files
selected_file = st.sidebar.selectbox("検索対象のファイルを選択:", file_options)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("キーワードを含めて質問してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if selected_file == "すべてのファイル":
        target_docs = all_docs
    else:
        target_docs = collection.get(where={"file_name": selected_file})["documents"]

    final_docs, context = search(
        prompt, target_docs, collection, selected_file, reranker
    )

    system_prompt = f"""あなたは誠実で優秀な社内AIアシスタントです。
提供された【参考情報】のみに基づいて、ユーザーの質問に正確に答えてください。

重要な指示:
- 情報が無い場合は、嘘をつかず「記載がありません」と答えること。
- 複数の資料で内容が食い違う場合は、無理に1つにまとめず、「資料Aでは〜、資料Bでは〜」と両方を併記し、どちらが正しいか断定しないこと。
- 数値（文字数・日数など）は、どの資料に基づくかが分かるように答えること。

【参考情報】
{context}"""

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in response:
            full_response += chunk["message"]["content"]
            placeholder.markdown(full_response + "▌")
        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    with st.expander("🔍 リランク後の採用資料（裏側）"):
        if final_docs:
            for score, doc in final_docs:
                st.markdown(f"* `[Score: {score:.3f}]` {doc}")
        else:
            st.markdown("採用された資料はありません。")
