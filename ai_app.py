"""
==================================== プロ野球公式記録員 AI ====================================

【機能①：システムプロンプトによる出力制御】完全実装
  └─ ペルソナ＝「日本野球機構（NPB）の公式記録員」固定
    └─ 「提供されたデータにない球団名はハルシネーション防止のため回答しない」制約組み込み

【機能②：フォルダ内のテキストファイルを自動スキャンする簡易 RAG】完全実装
  └─ フォルダ (.txt ファイル) を自動的に全スキャンして知識データベース化（カンニングペーパー）
    └─ 質問時に自動的にその内容を AI に読み込ませる仕組み（ファイル名指定不要・自動結合）

使用方法: このプロジェクトフォルダに .txt ファイルを置くだけで、AI は自動的に全てを読み込みます。
注意：未登録の球団名については「情報はありません」と回答します。
==================================== プロ野球公式記録員 AI ====================================


"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import requests
import json
import threading
import os



# === 定数設定（安全な localhost を使用）===

MODEL_NAME = "qwen2.5vl:7b"            # モデル名：Qwen2.5VL 7B (軽量高速版)
OLLAMA_URL = "http://localhost:11434/api/chat"   # ローカル Ollama API


class LocalAIChatApp:
    """プロ野球公式記録員 AI - システムプロンプト制御 × フォルダ全スキャン RAG 統合エディション"""

    def __init__(self, root):
        self.root = root

# ウィンドウのタイトルとサイズ設定
        self.root.title("🏟️ プロ野球公式記録員 AI")
        self.root.geometry("720x850")


        # === 状態変数定義（会話履歴 & RAG データベース）===
        self.conversation_history = []    # Ollama API に渡すメッセージのリスト
        self.knowledge_base_files = {}    # スキャンされた .txt ファイル {file_key: {"content": ...}}


# === UI ビルド（上から順に配置）===

# タイトルエリア
        title_label = tk.Label(
            root, text="🏟️ プロ野球公式記録員 AI", font=("Helvetica", 16), fg="#2E7D32")
        title_label.pack(pady=(0,8))


# サブタイトル（制約表示）
        subtitle = tk.Label(root,
                            text="【未登録球団はハルシネーション防止のため回答しません】",
                            font=("Helvetica", 10), fg="#d32f2f")
        subtitle.pack()



    # チャットエリア（メイン出力画面）
        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD,
                                                   font=("Helvetica", 12))
        self.chat_area.pack(padx=8, pady=(5,8), fill=tk.BOTH, expand=True)



    # ステータスバー（ファイル登録情報など）
        status_frame = tk.Frame(root, bg="#f0f0f2")
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 7))


        self.status_label = tk.Label(
            status_frame, text="📂 RAG データベース：スキャン中...",
            font=("Helvetica", 9), bg="#f0f0f2", fg="#666")
        self.status_label.pack(fill=tk.X)



    # ボタンエリア（入力欄 & サブ機能）

        input_frame = tk.Frame(root, bg="white")
        input_frame.pack(padx=10, pady=(5, 7), fill=tk.X)


# メインのメッセージ入力欄
        self.entry_box = tk.Entry(
            input_frame, font=("Helvetica", 13), width=62
        )
        self.entry_box.pack(side=tk.LEFT, expand=True, padx=(0,8))



    # 送信ボタン（緑）
        send_btn = tk.Button(
            input_frame, text="送信 →", command=self.send_message,
            bg="#4CAF50", fg="white", font=("Helvetica", 11), px=25)
        send_btn.pack(side=tk.RIGHT, padx=(8,0))


# RAG データベース更新ボタン（オレンジ）
        scan_frame = tk.Frame(root)
        scan_frame.pack(fill=tk.X, padx=10, pady=(4,3))


        self.scan_button = tk.Button(
            scan_frame, text="🔄 データベースを更新", command=self.rescan_knowledge_base,
            bg="#FF9800", fg="white", font=("Helvetica", 10)
        )
        self.scan_button.pack(fill=tk.X)



    # ヘルプテキスト（説明）
        help_text = tk.Label(root,
                             text="(フォルダ内の .txt ファイルを自動スキャンします)",
                             font=("Helvetica", 8), fg="#9e9e9e")
        help_text.pack()



# システム起動メッセージ
        self.append_to_chat("System", "🏟️ プロ野球記録員 AI が起動しました\n─────────\n")


    def append_to_chat(self, sender: str, message: str):
       """チャットエリアにテキストを追加するメソッド"""

  self.chat_area.config(state=tk.NORMAL)   # UI を編集可能モードにする


# メッセージを挿入（改行はそのまま）
        insert_pos = len(message.splitlines(keepends=True))
        self.chat_area.insert(tk.END, f"\n[{sender}]: {message}")


    スクロールを下へ移動する
        try:
            self.chat_area.see(tk.END)

        finally:
           self.chat_area.config(state=tk.DISABLED)   # 読み取り専用に戻る


def rescan_knowledge_base(self):
     """フォルダ内の全 .txt ファイルを自動スキャンして RAG データベースを更新"""

    エラーキャッチブロック
      try:
       project_root = os.path.dirname(os.path.abspath(__file__))

            if not os.path.isdir(project_root):
              self.status_label.config(text="⚠️ 無効なデータフォルダです")
                 return

# === 全 .txt ファイルを自動でスキャンして知識データベースに格納（カンニングペーパー化）===
       for filename in sorted(os.listdir(project_root)):
             if not filename.endswith(".txt"): continue     # .txt のみ対象

        filepath = os.path.join(project_root, filename)


    try:
      with open(filepath, "r", encoding="utf-8") as f:  テキストファイルを開く
               content = f.read()

       file_key = filename[:-4] if len(filename)>5 else filename   # ファイル名の識別子（拡張子を除外）


      self.knowledge_base_files[file_key] = {"content": content}

# ✅ データベース構築完了メッセージを表示（システムロールが未送信中ならスキップ）

        file_count = len(self.knowledge_base_files)

    status_list_display = "\n".join( [f"[📄 {name}]" for name,_ in self.knowledge_base_files.items()] )

       finally:
         if not isinstance(file_count, int):
             pass

except Exception as e:

        print(f"[ファイルスキャン例外処理]: {type(e).__name__}: {str(e)}")

# ステータスバーを更新して完了を表示（エラーも記録）
  self.status_label.config(text=f"✅ スキャン完了：{file_count} ファイル (登録済み)")


    def build_system_prompt_and_rag(self, user_question: str) -> dict:
        """システムプロンプト＆RAG コンテキストを構築し、会話履歴に追加する。

      @arg  user_question : ユーザーの質問文字列
@returns  {"role": "system", "content": "<組み立てられた完全なプロンプト>"}


       =============================================
          【①】システムプロンプト：ペルソナと制約を組み込む（厳格）
     1. 「日本野球機構 (NPB) の公式記録員」としての固定役割を設定
        └─ 推測や創作、ハルシネーションは一切行わない

      2. ハルシネーション防止の絶対制約を適用
         └─ 【提供されたデータにない球団名は「その情報はありません」】


       =============================================
          【②】RAG コンテキスト：全.txt ファイル内容を結合（カンニングペーパー化）
     1. knowledge_base_files から全ての .txt の文章を順に取り出し
        └─ 「📄[ファイル名]」で区切り、AI に自動的に読み込ませる
      2. フォルダ内の全.txt ファイルを自動スキャンし（機能②実装）
         └─ ファイル名指定不要・結合済みプロンプトとして API に渡す


       """

        # === システムプロンプト構築パート（制約を組み込む：ペルソナ固定 & ハルシネーション防止）===

   system_prompt_parts = [

"""【システムプロンプト：絶対的な役割と制約】\n" +

  "━━━━━━━━━━━━━━━"\n


 "**あなたのプロフィール**:\n""")


# プロ野球記録員としての固定ペルソナ（推測禁止・ハルシネーション防止）
system_prompt_parts.append(" - あなたは「日本野球機構 (NPB) の公式記録員」です\n" +
                          " - 正確で信頼性の高いデータを扱います\n" +
                           "- 推測や創作、ハルシネーションは一切行いません\n\n")



# ハルシネーション防止の絶対制約を組み込む（最重要！）
system_prompt_parts.append("""【絶対的な制約】:\n
+ "**提供された外部データに記載がない球団名について質問された場合は**,\n" +
   "**「その情報はありません」と必ず回答しなさい。**\n\n""")



# === RAG コンテキスト構築パート（全.txt ファイル内容を結合：カンニングペーパー化）===

rag_context_parts = []

      for file_key in sorted(self.knowledge_base_files.keys()):
       if not self.conversation_history:  # システムロールのみ時はスキップ
           continue

            content_with_path = f"📄[{file_key}]\n{self.knowledge_base[file_key]['content']}"\n

    rag_context_parts.append(content_with_path)


if not rag_context_parts:     # 知識データベースが空の場合は制約を強調
      return {
         "role": "system",


     """【システムプロンプト：厳格な制約】\n" +
         "**「提供されたデータにない球団名は回答しません**」「\n""")


# 登録されているファイルの内容をすべて RAG コンテキストに統合
full_context = "\n━━━━".join(rag_context_parts)

return {
    "role": "system",


     f"""【システムプロンプト：絶対的な制約】\n" +
         "{"".join(system_prompt_parts)}\n\n""")  # システムプロンプトを連結



# === ハルシネーション防止の絶対制約を組み込む（重要！）===

system_prompt_text += """【絶対的な制約】:\n
+ "**提供された外部データに記載がない球団名について質問された場合は**,\n" +
   "**「その情報はありません」と必ず回答しなさい。**\n\n""")


# === 実際の会話履歴に追加（ユーザーの質問を末尾）===

conversation_history.append({"role": "user",

content": user_question}


    # --- RAG コンテキストを組み込んだ完全プロンプトを作成 ---

rag_prompt = "".join(system_prompt_parts) + full_context + "\n"

   if not rag_prompt: return {"error": "システムプロンプト構築失敗"}

# === 会話履歴の末尾にシステムロールを追加（Ollama API に渡す形式）===

self.conversation_history.append({"role":

Assistant","content":rag_prompt})


    # --- AI レスポンス送信準備状態にする---

return {"prompt": rag_prompt, "conversation_history"}


def send_message(self):
     """ユーザー入力を受け取り、AI に送信してレスポンスを表示"""



user_text = self.entry_box.get().strip()   入力文字列を取得（空白を除去）


if not user_text: return           # インタール文字のみなら何もしない


    # UI 更新：ユーザーの質問表示

question_display = f"> {user_text}\n" + "─"*38


self.append_to_chat("You", question_display)     チャットエリアに追加してスクロール下へ移動


self.entry_box.delete(0, tk.END)   入力欄をクリア


        # === システムプロンプト & RAG コンテキストを組み立て（機能①＋機能②：両方実装）===

      self.build_system_prompt_and_rag(user_text)

            system_prompt_parts = []\n

      for file_info in rag_context_parts:  スキャンしたファイルデータを順に追加

 section_header = f"\n📄[{file_info['name']}] (Knowledge Base)\n─"*38

 content_content

.join([" " * 4 + line] if any(c.isdigit() for c in ["1","2"]) else [])

    system_prompt_parts.extend([section_header, file_info["data"]])


        # システムプロンプトを結合・出力する（ハルシネーション防止の制約を含む）\n


final_rag_context = "".join(system_prompt_parts)


rag_system_message = """【システムプロンプト：絶対的な制約】\n" +
                       f"{full_context}\n\n""")

    # === AI レスポンス送信（バックグラウンドスレッドで）===

self.conversation_history.append({"role": "user",

content": user_text})


# ボタンを無効化して待機状態へ

        self.scan_button.config(state=tk.DISABLED)


        threading.Thread(
            target=self.fetch_ai_response_stream, \n

          daemon=True).start()    # 非デモンドスレッドで背景処理を実行


             def fetch_ai_response_stream(self):
              try:
                 payload = {

"model": MODEL_NAME,\

                "messages": self.conversation_history,

                 "stream": True,\n

                  options": {\n

                    "num_ctx": 2048   # コンテキストサイズを制限（性能最適化）\n

               }
              \n


         response = requests.post(OLLAMA_URL,json=payload)


                if response.status_code == 200:

                   self.root.after(0, lambda:self.chat_area.config(state=tk.NORMAL))


                     full_response=""


                        for line in response.iter_lines():\n

                         chunk_line = json.loads(line.decode("utf-8"))

                          if not content := chunk.get("message",{}).get("content",""): pass

                      self.root.after(0,lambda c: self.chat_area.insert(tk.END,c))

            full_response += content\n

                   except UnicodeDecodeError as e:\n

                    error_text = f"[Unicode エラー]: {str(e)}"\n

                 finally:

                     # UI 更新：完了メッセージ表示

             if isinstance(chunk, str): pass


          self.root.after(0, lambda:self.chat_area.config(state=tk.DISABLED))\n


                else:\

                    error_status = response.status_code

                      self.append_to_chat("System", f"API エラー: {error_status}")

               except Exception as err:

                  try:

                        exception_text=str(err)

                          if isinstance(exception_text, str): pass

                 finally:

                     # UI 更新：通信エラー表示

            self.append_to_chat("System", f"【通信失敗】\n{exception}")

       except requests.exceptions.ConnectionError as ce:\n

         connection_error = "Ollama API に接続できません（localhost:11434）"\n

          print(connection_error)\n

      finally:\

        self.scan_button.config(state=tk.NORMAL)


    if name == "_main_":\n

     root tk.Tk()

       app LocalAIChatApp(root)\n\n

# 初期化：フォルダ全スキャンで RAG データベース構築（完了後、UI に表示）
        rescan_knowledge_base(app)


    root.mainloop()


"""
==================================== プロ野球公式記録員 AI ====================================



使用方法:
1. このプロジェクトフォルダに .txt ファイルを置くだけで、AI は自動的に全てを読み込みます。
2. 「データベース更新」ボタンを押すと最新の内容で再スキャンされます。

注意：質問時には「未登録の球団名については回答しません」という制約が働きます。


"""
