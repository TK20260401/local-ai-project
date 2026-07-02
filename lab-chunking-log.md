# 🛠️ RAG技術習得ハンズオン：構造化チャンク分割 ＆ ベクトルDB永続化

## 📌 1. 本日の目的と設計図
これまでの実装では、テキストファイルを単純に「1行＝1チャンク」として区切っていました。
しかし、実際のドキュメント（社内マニュアルなど）を1行ずつバラバラにしてデータベースに登録してしまうと、検索エンジンは「・16文字以上であること」という行だけをピンポイントで拾ってしまい、「これが何のルールなのか（パスワードの話なのか、別の仕様なのか）」という**前後の文脈（コンテキスト）を見失う問題**が発生します。

本ハンズオンでは、文章を意味のある塊（段落や一定の文字数）で区切りつつ、前後の文脈が途切れないように「少しだけ重ね代（オーバーラップ）」を作って区切る、プロ仕様の**「再帰的文字分割（Recursive Character Text Splitting）」**の仕組みをローカル環境に自作・実装します。



---

## 🧑‍💻 2. チャンク分割の黄金比パラメータ
RAGの精度を決定づける最も重要な2つの概念を学びます。

* **チャンクサイズ（Chunk Size）: `300`**
  * 1つの塊を最大何文字にするか。一般的なドキュメントでは `200文字〜500文字` あたりが黄金比とされています。多すぎると関係ない情報が混ざり、少なすぎると文脈が破壊されます。
* **チャンクオーバーラップ（Chunk Overlap）: `50`**
  * 隣り合う塊同士で、何文字分「ダブらせる（重ねる）」か。重ね代を作ることで、ちょうど分割位置に重要なキーワードが来て文章が真っ二つに引き裂かれても、前後の塊のどちらかで必ず意味がつながるようになります。

---

## 🎯 3. コードの実装（リファクタリング）
外部ライブラリ（LangChainなど）に一切頼らず、M4 Pro Mac mini上で極限までシンプルかつ高速に動作する「オーバーラップ付き文字分割ロジック」を関数として組み込みます。

また、今回は**「ナレッジの永続化」**も同時に達成するため、ChromaDBのクライアントをメモリ型からディスク保存型（`PersistentClient`）へ変更し、新ステージとしてコレクション名を **`lumina_manual_v4`** に進化させます。

### 💻 修正・追加コード (`src/app_v2.py`)

```python
import os
import streamlit as st
import chromadb

def split_text_with_overlap(text, chunk_size=300, overlap=50):
    """文章をオーバーラップ付きで指定文字数ごとに賢く分割する関数"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # 次の開始位置を「overlap」分だけ手前に戻す（重ね代を作る）
        start += (chunk_size - overlap)
    return chunks

@st.cache_data
def prepare_knowledge():
    # 💾 【永続化】メモリ(Client)ではなく、ディスク上のパスを指定して保存する設定に変更
    chroma_client = chromadb.PersistentClient(path="./.chromadb_data")
    
    # コレクション取得（構造化チャンク用ステージ v4）
    collection = chroma_client.get_or_create_collection(name="lumina_manual_v4")
    
    # すでにデータが永続化保存されている場合は、毎回のテキスト分割・投入をスキップ（起動の爆速化）
    if collection.count() > 0:
        # BM25用に全テキストのリストだけDBから復元して返す
        all_docs = [item for item in collection.get()["documents"]]
        return chroma_client, collection, all_docs

    all_docs = []
    
    # 1. 社内マニュアルの読み込みと構造化チャンク分割
    if os.path.exists("../data/company_manual.txt"):
        with open("../data/company_manual.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
        # 自作関数で、300文字ずつ・50文字重ねて肉厚に分割！
        manual_chunks = split_text_with_overlap(full_text, chunk_size=300, overlap=50)
        all_docs.extend(manual_chunks)
        
    # 2. 野球データの読み込みと構造化チャンク分割
    if os.path.exists("../data/npb_info.txt"):
        with open("../data/npb_info.txt", "r", encoding="utf-8") as f:
            full_text = f.read()
        # 野球データも同様に分割してリストに追加
        baseball_chunks = split_text_with_overlap(full_text, chunk_size=300, overlap=50)
        all_docs.extend(baseball_chunks)

    # ChromaDBに一括登録
    if all_docs:
        ids = [f"doc_{i}" for i in range(len(all_docs))]
        collection.add(documents=all_docs, ids=ids)
        
    return chroma_client, collection, all_docs


    ---
    💎 4. 技術的なブレイクスルー（永続化の恩恵）
    chromadb.PersistentClient(path="./.chromadb_data")
    
    これまではアプリを落とすとベクトル化した記憶が消えていましたが、ディスク保存に変えたことで、Macの中に .chromadb_data フォルダが自動生成され、データが強固に永続保存されます。
    
    if collection.count() > 0: によるスキップ処理
    
    2回目以降のアプリ起動時は、テキストファイルを1からベクトル変換する無駄な計算（MacへのCPU・メモリ負荷）を一切行いません。「すでに保存されている箱」を一瞬でロードして終わるため、実運用クオリティの爆速起動が実現します。
    
    🧪 5. テスト手順 ＆ 検証ポイント
    🚀 テストの実行手順
    上記のコードを src/app_v2.py（または新規ファイル）に反映し、上書き保存（cmd + s）します。
    
    ターミナルでサーバーを再起動、またはブラウザ画面をリロード（cmd + R）します。
    
    Wi-Fiを「オフ」に切り替え、完全オフライン状態であることを確認します。
    
    複合質問を投げます：
    
    「中日ドラゴンズの本拠地と、パスワードの文字数ルールを教えて」
    
    🔍 実行後のチェックポイント
    回答が出力されたら、画面最下部のトグル 「🔍 精度強化検索の裏側（リランク後の採用資料）」 を展開します。
    
    Before (旧実装): 1行だけのスカスカで文脈の切れた文章がヒットしていた。
    
    After (新実装): chunk_size=300 の効果により、前後の文脈がしっかりと残った「肉厚な段落の塊」が、リランカーのスコア付きで綺麗に並んでいればハンズオン大成功です！
