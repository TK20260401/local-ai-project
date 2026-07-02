# 🛡️ 1. 環境変数の盾
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 📦 2. 必要なライブラリのインポート
import chromadb
import docx
import ollama
import streamlit as st
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


# 🧠 3. リランカー（門番）のローカルロード
@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")


reranker = load_reranker()


# 🧼 4. 構造化チャンク分割ロジック
def split_text_with_overlap(text, chunk_size=300, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# 📄 5. テキスト抽出ロジック（PDF/Word）
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception:
        return ""


def extract_text_from_docx(docx_path):
    try:
        doc = docx.Document(docx_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text.append(paragraph.text)
        return "\n".join(text)
    except Exception:
        return ""


# 📂 6. 全自動インジェスト ＆ メタデータ永続化パイプライン
@st.cache_resource
def prepare_knowledge():
    # 💾 ディスク永続化（メタデータ対応のため新ステージ v8 へ移行！）
    chroma_client = chromadb.PersistentClient(path="./.chromadb_data")
    collection = chroma_client.get_or_create_collection(name="lumina_manual_v8")

    # すでに永続化DBにデータがある場合は、そこから全ドキュメントとメタデータを爆速ロード
    if collection.count() > 0:
        stored_data = collection.get()
        all_docs = stored_data["documents"]
        all_metadatas = stored_data["metadatas"]
        return chroma_client, collection, all_docs, all_metadatas

    all_docs = []
    all_metadatas = []
    data_dir = "../data"

    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            file_text = ""

            if filename.endswith(".pdf"):
                file_text = extract_text_from_pdf(file_path)
            elif filename.endswith(".docx"):
                file_text = extract_text_from_docx(file_path)
            elif filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    file_text = f.read()

            if file_text.strip():
                chunks = split_text_with_overlap(file_text, chunk_size=300, overlap=50)
                for chunk in chunks:
                    all_docs.append(chunk)
                    # 🟢 【タスク1-A】各チャンクに「元のファイル名」をタグ付けして記憶させる
                    all_metadatas.append({"file_name": filename})

    # ChromaDBに永続化登録
    if all_docs:
        ids = [f"doc_{i}" for i in range(len(all_docs))]
        dummy_embeddings = [[0.0]] * len(all_docs)

        # 🟢 ドキュメント、ID、ダミーベクトルと一緒に「メタデータ」を一括登録
        collection.add(
            documents=all_docs,
            ids=ids,
            embeddings=dummy_embeddings,
            metadatas=all_metadatas,
        )

    return chroma_client, collection, all_docs, all_metadatas


# ナレッジの準備
chroma_client, collection, all_docs, all_metadatas = prepare_knowledge()

# 🎨 7. Streamlit UI 描画ロジック
st.title("🎯 Lumina 超精度RAGチャット (メタデータ・フィルタリング搭載版)")

# 🟢 【タスク1-B】サイドバーにファイル絞り込み用のドロップダウンUIを設置
st.sidebar.header("📁 検索対象のフィルタリング")
# 重複のないファイル名リストを作成
unique_files = list(set([m["file_name"] for m in all_metadatas]))
# 選択肢に「すべてのファイル」を追加
file_options = ["すべてのファイル"] + unique_files
selected_file = st.sidebar.selectbox(
    "検索対象のファイルを選択してください:", file_options
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 🔍 8. メタデータ駆動型・ハイブリッド検索 ＆ リランク処理
if prompt := st.chat_input("キーワードを含めて質問してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 🟢 【タスク1-C】ユーザーのUI選択に応じて、検索対象の母集団を動的に切り替える
    if selected_file == "すべてのファイル":
        # 全件を対象にする
        target_docs = all_docs
    else:
        # ChromaDBから、選択されたファイル名のドキュメントだけを高速に物理抽出する！
        filtered_data = collection.get(where={"file_name": selected_file})
        target_docs = filtered_data["documents"]

    # 選択された母集団に対して、その場でBM25のインデックスを構築（ゴミデータの完全パージ）
    if target_docs:
        tokenized_docs = [list(doc) for doc in target_docs]
        bm25 = BM25Okapi(tokenized_docs)

        # BM25キーワード検索（最大15件）
        bm25_scores = bm25.get_scores(list(prompt))
        top_k = min(15, len(target_docs))
        top_bm25_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:top_k]
        candidate_docs = [target_docs[i] for i in top_bm25_indices]

        # CrossEncoderによる超精度リランキング
        pairs = [[prompt, doc] for doc in candidate_docs]
        rerank_scores = reranker.predict(pairs)
        scored_docs = sorted(
            zip(rerank_scores, candidate_docs), key=lambda x: x[0], reverse=True
        )

        # 上位5件を最終採用
        final_docs = scored_docs[:5]
        context = "\n---\n".join([doc for score, doc in final_docs])
    else:
        final_docs = []
        context = "指定されたファイルにデータがありません。"

    # 🤖 9. LLM（Ollama Qwen）へのプロンプト組み立てとストリーミング出力
    system_prompt = f"""あなたは誠実で優秀な社内AIアシスタントです。
提供された【参考情報】のみに基づいて、ユーザーの質問に正確に答えてください。
情報が足りない場合や、記載がない場合は、嘘をつかずに「記載がありません」と答えてください。

【参考情報】
{context}"""

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        response = ollama.chat(
            model="qwen3.5-9b-16k:latest",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )

        for chunk in response:
            full_response += chunk["message"]["content"]
            response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # 🔍 10. デバッグ用：裏側表示
    with st.expander("🔍 精度強化検索の裏側（リランク後の採用資料）"):
        if final_docs:
            for score, doc in final_docs:
                st.markdown(f"* `[Score: {score:.3f}]` {doc}")
        else:
            st.markdown("採用された資料はありません。")
