# 🚀 【1時間ラボ】完全オフライン・ハイブリッドRAG ── 精度極限突破 (BM25 × Chroma × Re-ranker)

> ルートCの核となる「ハイブリッド検索による取りこぼし防止」と「ローカル・リランカーによる順位最適化」を、1時間で手を動かして体得する実装ラボ。
> 意味で探すベクトル検索の弱点（キーワードの完全一致に弱い）と、LLMの弱点（情報が真ん中に挟まれると無視する現象）を、完全オフライン環境で同時に決着させる。

---

## 🎯 このラボで作るもの / 得られるもの

* **ハイブリッド検索（BM25 × ベクトル）**: 「意味が近い断片」と「特定のキーワード・固有名詞」を同時に網羅する最強の検索網。
* **ローカル・リランカー（再ランキング）**: 抽出した上位候補を、質問に本当に関連している順に並び替える軽量専門AI。
* **ストリーミングUIとの完全融合**: ルートBで構築したStreamlit画面に、ルートCの爆速かつ超高精度な脳細胞を移植する。

---

## ⏱️ タイムテーブル（60分）

| 時間 | やること | 対応スキル / コンポーネント |
| --- | --- | --- |
| 0–10分 | 準備（精度強化ライブラリの導入・モデル確認） | 環境構築（rank_bm25 / sentence-transformers） |
| 10–20分 | ナレッジベースの高度化（複数ファイルの整理） | 複数ドキュメントの配置（`data/`） |
| 20–45分 | `app_v2.py` の実装（ハイブリッド＆リランク） | Python（Streamlit / Chroma / BM25 / Re-ranker） |
| 45–55分 | オフライン検証（固有名詞・Lost in the Middle対策） | 精度検証テスト |
| 55–60分 | サーバー停止 ＆ Gitへバージョン管理プッシュ | バージョン管理（`app_v2.py`） |

---

## STEP 0（0–10分）：準備

Zedの内蔵ターミナル（`src` フォルダ内）を開き、精度強化に必要なライブラリを導入します。

```bash
# ハイブリッド検索用のBM25アルゴリズムと、ローカルリランカー用のライブラリを導入
pip install rank_bm25 sentence-transformers

# Ollamaに必要なモデルが揃っているか確認
ollama list
# 埋め込み用：mxbai-embed-large / 会話用：qwen3.5-9b-16k:latest

```

> **解説**: `rank_bm25` はGoogleなどの検索エンジンでも使われる「キーワードの完全一致」を秒速でスコア化するアルゴリズムです。`sentence-transformers` はMacのM4 Proチップ上で直接リランカーモデルを高速駆動させるために使用します。

---

## STEP 1（10–20分）：ナレッジベースの配置

すでに前回のディレクトリ構造案に基づいて、`data/` フォルダに必要なドキュメントが配置されているか確認します。

```text
local-ai-project/
├── data/
│   ├── company_manual.txt (社内パスワード等のルール)
│   └── npb_info.txt (球団・本拠地データ)
└── src/
    └── app.py (ルートBのソースコード)

```

> 固有名詞（「読売ジャイアンツ」「バンテリンドーム」など）や、記号・英数字（「16文字以上」「@」「#」など）が含まれるテキストを用意することで、ハイブリッド検索の圧倒的な強みを体感できます。

---

## STEP 2（20–45分）：精度強化版RAGアプリの実装

Zedのプロジェクトパネルから、`src` フォルダ直下に **`app_v2.py`** を新規作成（`src` を選択して `a` ➔ `app_v2.py`）し、以下のソースコードを貼り付けて保存（`cmd + s`）します。

```python
# src/app_v2.py ── 精度強化版ハイブリッドRAG (BM25 × Chroma × Local Re-ranker)
import streamlit as st
import ollama
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# 1. 各種クライアントと精度強化モデルの初期化
client = ollama.Client(host="http://127.0.0.1:11434")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="lumina_manual_v2")

# ローカル・リランカーモデルのロード（M4 Pro上で超高速に動作する軽量モデル）
@st.cache_resource
def load_reranker():
    return CrossEncoder("BAAI/bge-reranker-base")

reranker = load_reranker()

# UIタイトル
st.title("🎯 Lumina 超精度RAGチャット (ルートC：ハイブリッド＆リランク)")

# --- [裏舞台：データ投入] ---
# 実際の運用では一度だけ実行しますが、ラボ用に起動時チェックを行います
@st.cache_data
def prepare_knowledge():
    # 2つの資料からテキストを読み込んで合体
    documents = []
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

    # BM25用に文字をスペースで簡易トークナイズ（簡易日本語分かち書き）
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

        # ── 精度強化アルゴリズム開始 ──

        # 1. ベクトル検索（意味の近さで上位5件抽出）
        query_embed = client.embeddings(model="mxbai-embed-large", prompt=query)["embedding"]
        v_results = collection.query(query_embeddings=[query_embed], n_results=5)
        vector_hits = v_results["documents"][0] if v_results["documents"] else []

        # 2. BM25キーワード検索（完全一致で上位5件抽出）
        tokenized_query = list(query)
        bm25_hits = bm25_index.get_top_n(tokenized_query, all_docs, n=5)

        # 3. 検索結果の融合（重複を除去して候補群を作る）
        candidates = list(set(vector_hits + bm25_hits))

        # 4. ローカル・リランカーによる再ランキング（AIが質問と資料の関連度を厳密に採点）
        pairs = [[query, candidate] for candidate in candidates]
        scores = reranker.predict(pairs)

        # スコアが高い順に並び替え、最終的にLLMに渡す最強の上位3件（Top-3）を厳選
        scored_candidates = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        final_context_chunks = [doc for doc, score in scored_candidates[:3]]

        # ── 精度強化アルゴリズム終了 ──

        # プロンプトの組み立て
        retrieved_chunk = "\n\n".join(final_context_chunks)
        prompt = f"以下の【参考情報】に基づいて、質問に日本語で正確に答えてください。\n\n【参考情報】\n{retrieved_chunk}\n\n【質問】\n{query}\n\n【回答】"

        # ストリーミング出力
        full_response = ""
        for chunk in client.generate(model="qwen3.5-9b-16k:latest", prompt=prompt, stream=True):
            full_response += chunk['response']
            message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

        # デバッグ用：どの資料が選ばれたかを画面下に表示
        with st.expander("🔍 精度強化検索の裏側（リランク後の採用資料）"):
            for doc, score in scored_candidates[:3]:
                st.write(f"・`[Score: {score:.3f}]` {doc}")

```

---

## STEP 3（45–55分）：オフライン起動と2つの検証テスト

Wi-Fiを完全に「オフ」にし、Zedのターミナル（`src` フォルダ内）から新バージョンを起動します。

```bash
streamlit run app_v2.py

```

### 🧪 テストA：キーワード完全一致テスト（固有名詞）

> **質問**: 「バンテリンドームの本拠地はどこ？」 または 「中日ドラゴンズの本拠地は？」

* 従来のベクトル検索だけだと、「バンテリンドームナゴヤ」という特有の長い固有名詞のベクトルが上手くヒットしないことがありました。
* ハイブリッド検索の導入により、**BM25が「バンテリンドーム」という文字そのものを強烈にフック**し、完璧なコンテキストをLLMに渡すため、回答の初速と正確性が劇的に向上します。

### 🧪 テストB：Lost in the Middle（情報埋没）回避テスト

> **質問**: 複数の球団情報が混ざるような複雑な質問を投げます。

* 画面下の `🔍 精度強化検索の裏側` を開いてみてください。
* リランカーモデル（`bge-reranker-base`）が、大量の候補の中から「本当に質問の答えになっている一文」を**高スコアで1番上に強制ソート**しているのが確認できます。
* LLMのコンテキストの先頭に最も重要な情報が配置されるため、LLMの「真ん中の情報を無視する癖」を完全に封じ込めることができます。

---

## STEP 4（55–60分）：サーバー停止とGitへプッシュ

検証が完了したら、Zedのターミナルで **`ctrl + C`** を押してStreamlitサーバーを停止します。
その後、新しく作成した `app_v2.py` とこのラボドキュメントを含めて、成果をGitHubへ安全にプッシュします。

```bash
# 1. ルートフォルダに戻る
cd ..

# 2. 新ファイルをステージング
git add .

# 3. コミットメッセージを書いて保存
git commit -m "feat: ルートC 精度強化（ハイブリッド検索＆ローカルリランカー）の実装完了"

# 4. GitHubへ送信
git push

```

---

## 🧠 このラボで腑に落ちること（学びの言語化）

1. **ハイブリッド検索の二重網**:
ベクトル検索は「人間味のある曖昧な質問（〜についての概念など）」を拾い、BM25は「機械的な質問（型番、製品名、数値ルール）」を拾う。この2つの網を同時に投げることで、検索漏れが物理的にゼロになります。
2. **リランカーという「門番」**:
会話用LLM（Qwen）に大量の文章をそのまま読ませると、処理が重くなり「タメ」が悪化します。手前で軽量なリランカーに「点数付け」だけをさせ、本当に美味しい数件だけをLLMに渡すことで、**生成速度（タメの短縮）と回答精度の高次元での両立**が実現します。

---

## ❓ うまくいかないとき

| 症状 | 原因と対処 |
| --- | --- |
| `ModuleNotFoundError: rank_bm25` | ターミナルで `pip install rank_bm25 sentence-transformers` が未実行、または実行したターミナルの仮想環境がズレています。 |
| リランカーのロードでフリーズする | 初回起動時のみ、Hugging Faceからリランカーの軽量モデル（約400MB）をローカルに自動ダウンロードするため数十秒かかります。2回目以降は完全オフラインで一瞬で起動します。 |
| 応答の「タメ」が長くなった | 候補数を増やしすぎるとリランカーの計算が増えます。コード内の `n=5` や `n_results=5` の数値を `3` に減らすことで、M4 Proでの推論速度をさらに高速化できます。 |
