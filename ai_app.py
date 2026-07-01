import tkinter as tk
from tkinter import scrolledtext
import requests
import json
import threading
import os

# 【爆速化】熟考型の3.5:9Bから、打てば響く軽量な7Bモデルへ変更
MODEL_NAME = "qwen2.5vl:7b"
# 【手入力修正】ちぎれやすい数字ではなく、絶対に安全なlocalhostに書き換えます
OLLAMA_URL = "http://localhost:11434/api/chat"
DATA_FILE = "npb_data.txt"

class LocalAIChatApp:
    def __init__(self, root):
        self.root = root
        self.root.title("M4 Pro Max Speed RAG AI")
        self.root.geometry("600x700")
        self.root.geometry("600x700")

        self.conversation_history = []

        self.chat_area = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 12))
        self.chat_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)

        input_frame = tk.Frame(root)
        input_frame.pack(padx=10, pady=10, fill=tk.X)

        self.entry_box = tk.Entry(input_frame, font=("Helvetica", 14))
        self.entry_box.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.entry_box.bind("<Return>", lambda event: self.send_message())

        self.send_button = tk.Button(input_frame, text="送信", command=self.send_message, bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"))
        self.send_button.pack(side=tk.RIGHT)

        self.append_to_chat("System", f"【M4 Pro最適化・プチRAGモード起動: {MODEL_NAME}】\nフォルダ内の知識ファイルを自動で参照します。話しかけてください！\n" + "-"*50)

    def append_to_chat(self, sender, message):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"\n[{sender}]: {message}\n")
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def send_message(self):
        user_text = self.entry_box.get().strip()
        if not user_text:
            return

        self.append_to_chat("You", user_text)
        self.entry_box.delete(0, tk.END)

        # 【ハルシネーション改善：プチRAGの仕組み】
        # もしユーザーの質問に「NPB」や「球団」が含まれていたら、テキストファイル（カンニングペーパー）を読み込んでプロンプトに合体させる
        system_context = ""
        if os.path.exists(DATA_FILE) and ("NPB" in user_text.upper() or "球団" in user_text):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                kb_content = f.read()
            system_context = f"\n【重要情報（必ずこのデータに基づいて正確に答えてください）】\n{kb_content}\n"

        # AIに渡すメッセージの組み立て
        final_prompt = user_text
        if system_context:
            final_prompt = system_context + f"\nユーザーからの質問: {user_text}"

        self.conversation_history.append({"role": "user", "content": final_prompt})

        self.send_button.config(state=tk.DISABLED)
        threading.Thread(target=self.fetch_ai_response_stream, daemon=True).start()

    def fetch_ai_response_stream(self):
        try:
            payload = {
                "model": MODEL_NAME,
                "messages": self.conversation_history,
                "stream": True,
                "options": {
                    "num_ctx": 2048 # 記憶量を制限して初速を最速化
                }
            }
            
            response = requests.post(OLLAMA_URL, json=payload, stream=True)
            
            if response.status_code == 200:
                self.root.after(0, lambda: self.chat_area.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.chat_area.insert(tk.END, "\n[AI]: "))
                
                full_response = ""
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        content = chunk.get("message", {}).get("content", "")
                        full_response += content
                        self.root.after(0, lambda c=content: self.chat_area.insert(tk.END, c))
                        self.root.after(0, lambda: self.chat_area.see(tk.END))
                
                self.conversation_history.append({"role": "assistant", "content": full_response})
                self.root.after(0, lambda: self.chat_area.insert(tk.END, "\n"))
                self.root.after(0, lambda: self.chat_area.config(state=tk.DISABLED))
            else:
                status = response.status_code
                self.root.after(0, lambda: self.append_to_chat("System", f"エラー: {status}"))
        except Exception as err:
            error_msg = str(err)
            self.root.after(0, lambda: self.append_to_chat("System", f"通信失敗: {error_msg}"))
        finally:
            self.root.after(0, lambda: self.send_button.config(state=tk.NORMAL))

if __name__ == "__main__":
    root = tk.Tk()
    app = LocalAIChatApp(root)
    root.mainloop()
