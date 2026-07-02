import chromadb
import ollama
import streamlit as st

# OllamaクライアントとベクトルDB（Chroma）の準備
client = ollama.Client(host="http://127.0.0.1:11434")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="lumina_manual")

st.title("🤖 Lumina 社内RAGチャット (M4 Pro Local)")

# 1. ユーザーの入力を受け付けるチャットUI
if query := st.chat_input("マニュアルについて質問してください"):
    with st.chat_message("user"):
        st.markdown(query)

    # 2. RAGによる知識の検索（上位3つを結合）
    query_embed = client.embeddings(model="mxbai-embed-large", prompt=query)[
        "embedding"
    ]
    results = collection.query(query_embeddings=[query_embed], n_results=3)
    retrieved_chunk = "\n\n".join(results["documents"][0])

    # 3. プロンプトの組み立て
    prompt = f"以下の【参考情報】に基づいて、質問に日本語で正確に答えてください。\n\n【参考情報】\n{retrieved_chunk}\n\n【質問】\n{query}\n\n【回答】"

    # 4. ストリーミング（文字をパラパラ出す）で回答を表示
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # client.generate で stream=True を指定し、文字が生成されるたびに画面を更新
        for chunk in client.generate(
            model="qwen3.5-9b-16k:latest", prompt=prompt, stream=True
        ):
            full_response += chunk["response"]
            message_placeholder.markdown(
                full_response + "▌"
            )  # 入力中のカーソルエフェクト

        message_placeholder.markdown(full_response)  # 最終確定版を表示
