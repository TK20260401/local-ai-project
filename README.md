# local-ai-project

Mac mini（M4 Pro）単体・**完全オフライン**で動く、社内ドキュメント向けのローカルRAG（Retrieval-Augmented Generation）システム。

クラウドAPIに一切依存せず、社内資料を一切外に出さずに、PDF・Word・テキストを取り込んで「根拠付きで」質問に答える。文字化けするスキャンPDFはローカルOCRで読み取り、資料に無いことは正直に「記載がありません」と答える。

## 特徴

- **完全オフライン**: Wi-Fiを切った状態で起動・検索・回答まで完結。機密資料が外部に送られない。
- **ハイブリッド検索**: BM25（キーワード一致）× ベクトル検索（意味の近さ）で、固有名詞も曖昧な質問も取りこぼさない。
- **リランカーによる精度向上**: 候補をローカルの再ランキングモデルで並べ直し、本当に関連する資料だけをLLMへ渡す。
- **ローカルOCR**: 文字化けするPDFを画像として読み取り、テキスト化して検索対象にする。
- **ハルシネーション抑制**: 資料に無い情報は創作せず「記載がありません」と回答。資料間で矛盾する場合は出典付きで併記する。

## 技術スタック

| 役割 | 使用技術 |
|---|---|
| 推論エンジン | Ollama |
| 会話モデル | qwen3.5-9b-16k |
| 埋め込み | bge-m3 |
| リランカー | BAAI/bge-reranker-base |
| OCR | EasyOCR |
| ベクトルDB | ChromaDB（永続化） |
| UI | Streamlit |

## ディレクトリ構成

\`\`\`
```
local-ai-project/
├── data/
├── docs/
│   ├── knowledge/
│   └── archive/
└── src/
```

- `data/` — ナレッジ資料（.txt / .pdf / .docx）
- `docs/knowledge/` — 統合ナレッジ・新人タスクリスト
- `docs/archive/` — 過去の開発ログ・ハンズオン記録
- `src/` — アプリ本体（`app_v7_ocr.py` が最新）
\`\`\`

## セットアップ

### 1. Ollama とモデルの用意

[Ollama](https://ollama.com) をインストールし、必要なモデルを取得する。

\`\`\`bash
ollama pull qwen3.5-9b-16k   # 会話モデル
ollama pull bge-m3           # 埋め込みモデル
\`\`\`

取得済みのモデルは \`ollama list\` で確認できる。モデル名の表記（\`:latest\` の有無など）は環境により異なるため、コード側の指定と一致させること。

### 2. Python 依存ライブラリの導入

\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 動かし方

\`\`\`bash
# data/ に資料を置き、src へ移動して起動
cd src
streamlit run app_v7_ocr.py
\`\`\`

初回のみ、OCR・リランカーのモデル取得にネット接続が必要。一度取得すれば以降は完全オフラインで動作する。

## ドキュメント

- 設計・経緯・デバッグ記録: [\`docs/knowledge/local-ai-project_knowledge.md\`](docs/knowledge/local-ai-project_knowledge.md)
- 新メンバー向け育成タスク: [\`docs/knowledge/newcomer_tasklist.md\`](docs/knowledge/newcomer_tasklist.md)
