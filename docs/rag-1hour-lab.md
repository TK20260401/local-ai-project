# 【1時間ラボ】完全オフライン・ミニRAGを組む ── bge-m3 × Chroma × Ollama

> 5大スキルの「スキル2（出力構造化で嘘を封じる）」と「スキル3（ベクトルDBで本格RAG）」を、1時間で手を動かして体得する実装ラボ。
> 開発ログで実際に幻覚（「ハーレーズ・ホークス」）が出た **NPB球団名** を題材にして、「資料に無いことは答えない」AIを自分で作る。
> 作業は全部Zed内で完結させ、**3大プロ技（`/file`・Insert・New Thread）**も同時に練習する。

---

## 🎯 このラボで作るもの / 得られるもの

- ネットを切った状態で、テキスト資料をベクトル化してChromaに蓄積し、質問に**資料の根拠だけ**で答えるRAGツール。
- 効果を体感する2つのテスト：資料にある質問には正確に答え、**資料に無い質問には「記載がありません」と正直に答える**（＝幻覚の封じ込め）。
- これが動けば、あなたの手持ちの `bge-m3` が「埋め込みモデル」として何をしているか、Chromaが「意味の近い断片を検索する箱」として何をしているかが、体で分かる。

**練習する3大プロ技**：`/file`（コードをAIに見せて解説させる）／ Insert（AIの修正を1秒で反映）／ New Thread（重くなったら文脈リセット）。

---

## ⏱️ タイムテーブル（60分）

| 時間 | やること | 対応スキル/プロ技 |
|---|---|---|
| 0–10分 | 準備（ライブラリ導入・bge-m3確認） | 環境 |
| 10–20分 | 資料ファイル `npb_info.txt` を作る | スキル3の素材 |
| 20–45分 | `rag_local.py` を作る（Zedで） | スキル2・3／プロ技1・2 |
| 45–55分 | 実行して2つのテスト | 検証 |
| 55–60分 | New Threadで締め＋GitHubへpush | プロ技3 |

---

## STEP 0（0–10分）：準備

Zedを開き、ターミナル（Zed下部、または通常のターミナル）で以下を実行。

```bash
# 必要ライブラリを導入
pip install ollama chromadb

# 埋め込みモデルがあるか確認（無ければ pull）
ollama list
# 一覧に bge-m3 が無ければ:
ollama pull bge-m3
```

> `bge-m3` は「文章を意味の座標（ベクトル）に変換する」専用モデル。会話用モデルとは役割が違う。RAGの心臓部。

---

## STEP 1（10–20分）：資料ファイルを作る

プロジェクト内に `npb_info.txt` を作成。**1行 = 1チャンク（検索の最小単位）**になるので、各行が単体で意味を持つように書くのがコツ。

```text
セ・リーグの球団：読売ジャイアンツ。本拠地は東京ドーム。
セ・リーグの球団：阪神タイガース。本拠地は阪神甲子園球場。
セ・リーグの球団：中日ドラゴンズ。本拠地はバンテリンドームナゴヤ。
セ・リーグの球団：横浜DeNAベイスターズ。本拠地は横浜スタジアム。
セ・リーグの球団：広島東洋カープ。本拠地はMAZDA Zoom-Zoomスタジアム広島。
セ・リーグの球団：東京ヤクルトスワローズ。本拠地は明治神宮野球場。
パ・リーグの球団：福岡ソフトバンクホークス。本拠地はみずほPayPayドーム福岡。
パ・リーグの球団：北海道日本ハムファイターズ。本拠地はエスコンフィールドHOKKAIDO。
パ・リーグの球団：千葉ロッテマリーンズ。本拠地はZOZOマリンスタジアム。
パ・リーグの球団：埼玉西武ライオンズ。本拠地はベルーナドーム。
パ・リーグの球団：オリックス・バファローズ。本拠地は京セラドーム大阪。
パ・リーグの球団：東北楽天ゴールデンイーグルス。本拠地は楽天モバイルパーク宮城。
```

> これが「AIにカンニングさせる正解データ」。ここに書いていないことは、AIは答えられない設計にする（次のSTEP）。

---

## STEP 2（20–45分）：RAG本体を作る

`rag_local.py` を作成し、以下を貼り付ける。**まずは丸ごと動かし**、そのあとプロ技1で1関数ずつAIに解説させて理解する（後述）。

```python
# rag_local.py ── 完全オフライン・ミニRAG (bge-m3 × Chroma × Ollama)
import ollama
import chromadb

KB_FILE     = "npb_info.txt"
EMBED_MODEL = "bge-m3"            # 埋め込み専用モデル
CHAT_MODEL  = "qwen2.5-coder:7b"  # ollama list にある会話モデル名に合わせる
TOP_K       = 3                   # 検索で拾う件数


def embed(text: str) -> list[float]:
    """文章を bge-m3 でベクトル（数字の羅列）に変換する"""
    res = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return res["embedding"]


def load_chunks(path: str) -> list[str]:
    """資料を読み、空でない行をチャンク（検索の最小単位）にする"""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    return [line.strip() for line in raw.splitlines() if line.strip()]


def build_db(chunks: list[str]):
    """チャンクを全部ベクトル化して、メモリ上のChromaに登録する"""
    client = chromadb.Client()                       # 起動ごとに作り直す（学習用）
    col = client.get_or_create_collection("npb")
    embeddings = [embed(c) for c in chunks]           # ← ここでbge-m3が全部変換
    col.add(
        ids=[str(i) for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks,
    )
    return col


def search(col, question: str, k: int = TOP_K) -> list[str]:
    """質問をベクトル化し、意味が近いチャンクを上位k件だけ取り出す"""
    q_emb = embed(question)
    res = col.query(query_embeddings=[q_emb], n_results=k)
    return res["documents"][0]


def answer(question: str, context_chunks: list[str]) -> str:
    """拾った資料だけを根拠に、Ollamaの会話モデルに答えさせる（嘘防止）"""
    context = "\n".join(f"- {c}" for c in context_chunks)
    system = (
        "あなたは資料に忠実なアシスタントです。"
        "以下の【資料】に書かれている内容だけを根拠に、日本語で簡潔に答えてください。"
        "資料に無いことは絶対に推測・創作せず、その場合は「資料に記載がありません」とだけ答えてください。"
    )
    user = f"【資料】\n{context}\n\n【質問】\n{question}"
    resp = ollama.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return resp["message"]["content"]


def main():
    chunks = load_chunks(KB_FILE)
    col = build_db(chunks)
    print(f"✅ {len(chunks)}件のチャンクをローカルDBに登録（完全オフライン）")
    print("質問をどうぞ（終了は Ctrl+C）\n")
    while True:
        q = input("質問> ").strip()
        if not q:
            continue
        hits = search(col, q)
        print("\n[回答]")
        print(answer(q, hits))
        print("\n[参照した資料]")
        for h in hits:
            print(" -", h)
        print()


if __name__ == "__main__":
    main()
```

> ⚠️ `CHAT_MODEL` は `ollama list` に**実在する名前**に。無ければ `qwen3.5:9b` など手持ちの会話モデルに変える（`bge-m3` は埋め込み専用なのでここには入れない）。

### 🧩 ここで3大プロ技を練習する

- **プロ技1（`/file`）**：Zedのチャットに `/file rag_local.py この build_db 関数が何をしているか、素人にもわかるように解説して` と入力。コードを貼らずにAIへ丸ごと見せて理解を深める。
- **プロ技2（Insert）**：もしエラーが出たら、エラー文をチャットに貼り「直して」。AIの修正コードを **Insert** でその場上書き（コピペ事故ゼロ）。
- **プロ技3（New Thread）**：解説の話題を変えるとき、`New Thread`（＋アイコン）で文脈リセット。初速の「タメ」が消える。

---

## STEP 3（45–55分）：実行して2つのテストをする

Wi-Fiを切って（完全オフライン確認）、実行する。

```bash
python rag_local.py
```

### テストA：資料にある質問 → 正確に答える

```
質問> パ・リーグの球団を教えて
```
→ ソフトバンク・日本ハム・ロッテ・西武・オリックス・楽天が、資料どおり正確に返れば成功。開発ログの「ハーレーズ・ホークス」幻覚が、RAGで解消されたことになる。

### テストB：資料に無い質問 → 正直に「記載がありません」

```
質問> 各球団の年間観客動員数は？
```
→ **「資料に記載がありません」**と返れば大成功。これがスキル2（出力構造化で嘘を封じる）の効果。適当に数字をでっち上げないAIが、システムプロンプトの制約だけで実現できている。

---

## STEP 4（55–60分）：締めとpush

理解が済んだら、Zedのチャットは `New Thread` で締める（プロ技3）。成果をGitHubへ。

```bash
git add .
git commit -m "ミニRAG実装（bge-m3×Chroma）: スキル2・3の基礎完了"
git push
```

---

## 🚀 時間が余ったら（ストレッチ課題）

- **チャンクを賢くする**：1行単位ではなく、複数行をまとめた段落単位でチャンク化してみる（長文資料での精度を体感）。
- **TOP_K を変える**：`TOP_K = 1` と `5` で回答の変化を見る。拾いすぎるとタメが増える＝スキル3の計算量トレードオフを実感。
- **資料を差し替える**：`npb_info.txt` を自分の業務メモや技術ドキュメントに変えて、社内RAGの原型にする。
- **永続化**：`chromadb.Client()` を `chromadb.PersistentClient(path="./chroma_db")` に変え、起動のたびに再ベクトル化しないようにする（スキル3の実運用寄り）。

---

## 🧠 このラボで腑に落ちること（学びの言語化）

- **埋め込み（bge-m3）** ＝ 文章を「意味の座標」に変える。似た意味の文は座標が近い。
- **ベクトルDB（Chroma）** ＝ その座標で「近いもの」を高速に探す箱。全文スキャンより圧倒的に軽い＝タメが消える。
- **構造化プロンプト（スキル2）** ＝ 「資料の外は答えるな」と縛るだけで、幻覚が止まる。精度はモデルの賢さより設計で決まる。

この3点が、開発ログで直面した「タメ」と「嘘」の両方に、同時に決着をつける。次の本格RAG（永続化・大規模データ・LangChain化＝スキル4）への確かな一歩になる。

---

## ❓ うまくいかないとき

| 症状 | 対処 |
|---|---|
| `ModuleNotFoundError: ollama / chromadb` | `pip install ollama chromadb` を実行 |
| 埋め込みでエラー | `ollama.embeddings(...)` が古い場合、`ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]` に変える |
| モデル名エラー | `ollama list` の名前に `CHAT_MODEL` を合わせる |
| 応答が遅い | `New Thread`／`TOP_K` を下げる／会話モデルを軽量な `qwen2.5-coder:7b` に |
| 日本語ファイルが読めない | ファイルをUTF-8で保存し直す |
