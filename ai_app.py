import glob
import json
import threading
import tkinter as tk
from tkinter import scrolledtext

import requests

# ==============================================================================
# ⚾ プロ野球公式記録員 AI (完成形)
# ==============================================================================

# 【M4 Pro 爆速仕様】完全オフライン用のローカルモデルとアドレス
MODEL_NAME = "qwen3.5-9b-16k:latest"
OLLAMA_URL = "http://localhost:11434/api/chat"


class LocalAIChatApp:
    def __init__(self, root):
        # UI と状態管理の初期化
        self.root = root
        self.root.title("プロ野球公式記録員 AI")
        self.root.geometry("600x700")

        # 会話の履歴を保存するリスト
        self.conversation_history = []

        # チャット表示エリア
        self.chat_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=("Helvetica", 12)
        )
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)

        # 入力フレーム
        input_frame = tk.Frame(root)
        input_frame.pack(padx=10, pady=10, fill=tk.X)

        # テキスト入力欄
        self.entry_box = tk.Entry(input_frame, font=("Helvetica", 14))
        self.entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry_box.bind("<Return>", lambda event: self.send_message())

        # 送信ボタン
        self.send_button = tk.Button(
            input_frame,
            text="送信",
            command=self.send_message,
            bg="#009688",
            fg="white",
            font=("Helvetica", 12, "bold"),
        )
        self.send_button.pack(side=tk.RIGHT)

        self.append_to_chat(
            "System",
            f"【自動全スキャンRAG起動: {MODEL_NAME}】\nフォルダ内のすべての .txt ファイルを自動結合して読み込みます。\n"
            + "-" * 50,
        )

    def append_to_chat(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"\n[{sender}]: {message}\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def load_all_txt_files(self):
        """【機能②】フォルダ内のすべての .txt ファイルを自動スキャンして結合"""
        combined_knowledge = ""
        txt_files = glob.glob("*.txt")

        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    combined_knowledge += (
                        f"\n--- ファイル名: {file_path} ---\n{content}\n"
                    )
            except Exception as e:
                print(f"読み込み失敗 ({file_path}): {e}")

        return combined_knowledge

    def send_message(self):
        user_text = self.entry_box.get().strip()
        if not user_text:
            return

        self.append_to_chat("You", user_text)
        self.entry_box.delete(0, tk.END)

        # 1. フォルダ内のテキスト知識を自動全スキャンしてロード
        knowledge_base = self.load_all_txt_files()

        # 2. 【機能①】システムプロンプトによる絶対的制約（ペルソナ固定とハルシネーション防止）
        system_instruction = (
            "あなたは『プロ野球公式記録員』です。以下の【提供された外部データ】に記載されている事実のみを基準に、ユーザーの質問に日本語で正確に答えてください。\n"
            "データに記載がない球団名や情報については、ハルシネーション（嘘）を防ぐために、知ったかぶりをせず必ず『提供されたデータに情報がありません』と正直に回答してください。絶対に嘘を合成してはいけません。\n\n"
            f"【提供された外部データ】\n{knowledge_base}\n"
        )

        # 毎回最新のテキスト状態をシステムプロンプトとして上書き更新（文脈パニックの防止）
        self.conversation_history = [{"role": "system", "content": system_instruction}]

        # ユーザーの発言を履歴に追加
        self.conversation_history.append({"role": "user", "content": user_text})

        # 送信ボタンを無効化してバックグラウンドでAIの回答を待つ
        self.send_button.config(state=tk.DISABLED)
        threading.Thread(target=self.fetch_ai_response_stream, daemon=True).start()

    def fetch_ai_response_stream(self):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": self.conversation_history,
                "stream": True,
                "options": {
                    "num_ctx": 2048  # 【熟考対策】記憶長を絞り込み、最初のタメを短縮
                },
            }

            response = requests.post(OLLAMA_URL, json=payload, stream=True)

            if response.status_code == 200:
                self.root.after(0, lambda: self.chat_area.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.chat_area.insert(tk.END, "\n[AI]: "))

                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode("utf-8"))
                        content = chunk.get("message", {}).get("content", "")
                        full_response += content
                        self.root.after(
                            0, lambda c=content: self.chat_area.insert(tk.END, c)
                        )
                        self.root.after(0, lambda: self.chat_area.see(tk.END))

                self.conversation_history.append(
                    {"role": "assistant", "content": full_response}
                )
                self.root.after(0, lambda: self.chat_area.insert(tk.END, "\n"))
                self.root.after(0, lambda: self.chat_area.config(state=tk.DISABLED))
            else:
                status = response.status_code
                self.root.after(
                    0, lambda: self.append_to_chat("System", f"エラー: {status}")
                )
        except Exception as err:
            error_msg = str(err)
            self.root.after(
                0, lambda: self.append_to_chat("System", f"通信失敗: {error_msg}")
            )
        finally:
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))


if __name__ == "__main__":
    # UI の起動
    root = tk.Tk()
    app = LocalAIChatApp(root)
    root.mainloop()
