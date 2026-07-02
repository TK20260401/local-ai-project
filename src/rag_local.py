"""
RAG スクリプト - 完全オフライン対話ループ版。
フォルダ内のすべての .txt を自動読み込み、何回でも連続して質問が可能です。
"""

import json
import os

import ollama

# 使用するローカルAIモデルの名前とURL
MODEL_NAME = "qwen3.5-9b-16k:latest"
OLLAMA_URL = "http://localhost:11434"


def build_db(project_dir):
    """フォルダ内の .txt を自動読み込んで知識データベース化するメソッド"""
    kb_db = {}
    for filename in sorted(os.listdir(project_dir)):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(project_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            file_key = filename[:-4] if len(filename) > 4 else filename
            kb_db[file_key] = {"content": content}
        except Exception as e:
            print(f"[警告]: {filename} の読み込みに失敗しました: {e}")
    return kb_db


def build_system_prompt(knowledge_base, user_question):
    """システムプロンプト＆RAG コンテキストを構築"""
    system_prompt = (
        "【システム：NPB 公式記録員】\n\n"
        "**役割**: あなたは「日本野球機構 (NPB) の公式記録員」です。\n"
        " - 正確で信頼性の高いデータを扱います\n"
        " - ハルシネーション（推測・創作）は一切行いません\n\n"
        "【絶対制約】: 提供されたデータに記載がない球団名や情報について質問された場合は、"
        "必ず「その情報はありません」と回答しなさい。絶対に嘘を吐いてはいけません。\n"
        "────────────────\n"
    )

    rag_context_parts = []
    for file_key in sorted(knowledge_base.keys()):
        content = knowledge_base.get(file_key, {}).get("content", "")
        if content:
            content_with_path = f"📄【ファイル名: {file_key}.txt】\n{content}\n"
            rag_context_parts.append(content_with_path)

    full_context = (
        "\n━━━━\n".join(rag_context_parts)
        if rag_context_parts
        else "提供されたデータはありません。"
    )
    final_prompt = f"{system_prompt}\n【提供された外部データ】\n{full_context}\n━━━━\n質問: {user_question}"
    return final_prompt


def ask_local_ai(prompt):
    """構築したプロンプトをOllamaに投げて、ストリーミング出力で回答させる"""
    print("\n🔄 AIが思考中（完全オフライン）...")
    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        print("\n[AI]: ", end="", flush=True)
        for chunk in response:
            print(chunk["message"]["content"], end="", flush=True)
        print("\n")
    except Exception as e:
        print(f"\n[エラー] Ollamaとの通信に失敗しました: {e}")


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 1. 知識データベースの構築
    kb_db = build_db(project_root)
    print("✅ RAG データベース構築完了")
    print(f"登録ファイル数: {len(kb_db)}")
    for key, info in kb_db.items():
        print(f"  └── {key}.txt: {len(info['content'])} バイト")
    print("-" * 50)

    # 2. 【改善】無限対話ループの実装
    while True:
        # flush=True を追加して、入力待ちの文字が一瞬でターミナルに表示されるように修正
        print("質問を入力してください（『終了』でストップ）> ", end="", flush=True)
        user_question = input().strip()

        # 終了キーワードが打たれたらループを抜ける
        if user_question in ["終了", "しゅうりょう", "exit", "quit", "バイバイ"]:
            print("\n公式記録員AIのセッションを終了します。お疲れ様でした！")
            break

        if not user_question:
            continue

        # プロンプトの組み立てとAIへの送信
        rag_prompt = build_system_prompt(
            knowledge_base=kb_db, user_question=user_question
        )
        ask_local_ai(rag_prompt)
        print("-" * 50)
