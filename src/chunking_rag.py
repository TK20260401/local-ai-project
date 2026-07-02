import os

import chromadb
import ollama

OLLAMA_HOST = "http://127.0.0.1:11434"
client = ollama.Client(host=OLLAMA_HOST)


# ==========================================
# 1. ドキュメントの読み込みと「構造ベース」の分割（Chunking）
# ==========================================
def load_and_chunk_markdown(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} が見つかりません。")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Markdownの見出し「##」または「###」で分割を試みる
    lines = content.split("\n")
    chunks = []
    current_chunk = []

    for line in lines:
        # 新しい見出しが始まったら、そこまでの文章を1つのチャンクとして確定する
        if line.startswith("##") or line.startswith("###"):
            if current_chunk:
                chunks.append("\n".join(current_chunk).strip())
            current_chunk = [line]  # 新しいチャンクの開始（見出しを含める）
        else:
            if current_chunk or line.strip():  # 空行のみで始まらないように制御
                current_chunk.append(line)

    # 最後のチャンクを追加
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return chunks


# ==========================================
# 2. ベクトルDBへの登録とRAGの実行
# ==========================================
def main():
    # マニュアルをチャンクに分割
    # ⬇️ 修正後（プログラムから見て、1つ上の階層の data/ フォルダの中を見に行く設定）
    manual_path = "../data/company_manual.txt"
    chunks = load_and_chunk_markdown(manual_path)

    print(f"--- マニュアルを分割しました（全 {len(chunks)} チャンク） ---")
    for i, chunk in enumerate(chunks):
        print(
            f"\n[チャンク {i + 1} の先頭部分]:\n{chunk.splitlines()[0]} ... ({len(chunk)}文字)"
        )

    # ChromaDBの初期化
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="lumina_manual")

    # 各チャンクをベクトル化してDBに登録
    print("\n--- ベクトルDBにチャンクを登録中 ---")
    for i, chunk in enumerate(chunks):
        embed_res = client.embeddings(model="mxbai-embed-large", prompt=chunk)
        collection.add(
            ids=[f"manual_chunk_{i + 1}"],
            embeddings=[embed_res["embedding"]],
            documents=[chunk],
        )

    # ユーザーからの質問テスト
    query = "パスワードの変更頻度と、文字数のルールはどうなっていますか？"
    print(f"\nユーザーの質問: {query}")

    # 質問に最も関連する「1つのチャンク」だけをピンポイントで検索
    query_embed = client.embeddings(model="mxbai-embed-large", prompt=query)[
        "embedding"
    ]
    # 修正後：上位3つのチャンクを取得して、1つのテキストに結合する
    results = collection.query(query_embeddings=[query_embed], n_results=3)
    retrieved_chunk = "\n\n".join(results["documents"][0])  # 3つのチャンクを結合

    print(
        f"\n--- 🔍 検索されたピンポイントな関連情報 ---\n{retrieved_chunk}\n------------------------------------------"
    )

    # LLMへのプロンプト組み立て（Qwen3.5を使用）
    system_prompt = (
        "以下の【参考情報】に基づいて、ユーザーの質問に日本語で正確に答えてください。"
    )
    prompt = f"{system_prompt}\n\n【参考情報】\n{retrieved_chunk}\n\n【質問】\n{query}\n\n【回答】"

    print("\n--- AIが回答を生成中（M4 Proで推論） ---")
    response = client.generate(model="qwen3.5-9b-16k:latest", prompt=prompt)
    print(f"\n[AIの回答]:\n{response['response']}")


if __name__ == "__main__":
    main()
