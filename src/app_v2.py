# ==============================================================================
# 🟢 1. 【最重要】通信の強制シャットダウン設定（コードの最初に配置）
# ==============================================================================
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # 外部ネットワークへの生存確認を完全に禁止する命令

import chromadb
import ollama
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 2. 各種クライアントと精度強化モデルの初期化
client = ollama.Client(host="http://127.0.0.1:11434")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="lumina_manual_v3")


# 🟢 エラーの原因だった引数をすべて削除し、一番安全な形に戻しました
@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")


reranker = load_reranker()

st.title("🎯 Lumina 超精度RAGチャット (ハイブリッド＆リランク)")


# --- [裏舞台：データの準備と投入] ---
@st.cache_data
def prepare_knowledge():
    documents = []
    # 2つのテキストファイルからデータを読み込んで合体
    for path in ["../data/company_manual.txt", "../data/npb_info.txt"]:
        try:
            with open(path, encoding="utf-8") as f:
                documents.extend([line.strip() for line in f if line.strip()])
        except FileNotFoundError:
            pass

    # ChromaベクトルDBに登録
    for i, doc in enumerate(documents):
        emb = client.embeddings(model="mxbai-embed-large", prompt=doc)["embedding"]
        collection.add(ids=[str(i)], embeddings=[emb], documents=[doc])

    # BM25用に文字をバラしてトークナイズ（簡易分かち書き）
    tokenized_corpus = [list(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    return documents, bm25


all_docs, bm25_index = prepare_knowledge()

# --- [表舞台：対話UI] ---
if query := st.chat_input("キーワードや固有名詞を含めて質問してください"):
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        # ─── 精度強化検索アルゴリズム ───

        # ① ベクトル検索（意味の近さで上位5件抽出）
        query_embed = client.embeddings(model="mxbai-embed-large", prompt=query)[
            "embedding"
        ]
        # 🔍 修正対象の場所：
        # ① ベクトル検索（n_results を 5 ➔ 8 に増やす）
        v_results = collection.query(query_embeddings=[query_embed], n_results=8)
        vector_hits = v_results["documents"][0] if v_results["documents"] else []

        # ② BM25キーワード検索（nを5 → 8 に増やす）
        tokenized_query = list(query)
        bm25_hits = bm25_index.get_top_n(tokenized_query, all_docs, n=8)

        # ③ 検索結果の融合（重複を排除して候補群を作る）
        candidates = list(set(vector_hits + bm25_hits))

        # ④ ローカル・リランカーによる再ランキング（質問との関連度を厳密に採点）
        pairs = [[query, candidate] for candidate in candidates]
        scores = reranker.predict(pairs)

        # スコアが高い順に並び替え、最終的にLLMに渡す最終件数を絞る（[:3] ➔ [:5] に増やす）
        scored_candidates = sorted(
            zip(candidates, scores), key=lambda x: x[1], reverse=True
        )
        final_context_chunks = [doc for doc, score in scored_candidates[:5]]

        # ─── 精度強化検索アルゴリズム終了 ───

        # プロンプトの組み立て
        retrieved_chunk = "\n\n".join(final_context_chunks)
        prompt = f"以下の【参考情報】に基づいて、質問に日本語で正確に答えてください。\n\n【参考情報】\n{retrieved_chunk}\n\n【質問】\n{query}\n\n【回答】"

        # ストリーミング出力
        full_response = ""
        for chunk in client.generate(
            model="qwen3.5-9b-16k:latest", prompt=prompt, stream=True
        ):
            full_response += chunk["response"]
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        # デバッグ用：選ばれた根拠資料とスコアを画面下に表示
        with st.expander("🔍 精度強化検索の裏側（リランク後の採用資料）"):
            for doc, score in scored_candidates[:3]:
                st.write(f"・`[Score: {score:.3f}]` {doc}")
