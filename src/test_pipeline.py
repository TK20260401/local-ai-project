import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import numpy as np
import easyocr
import pdf2image
import ollama
import chromadb

PDF_PATH = "../data/travel_rule.pdf"
EMBED_MODEL = "bge-m3"

def log(msg, t0):
    print(f"[{time.time()-t0:.1f}s] {msg}", flush=True)

t0 = time.time()
log("start", t0)

log("OCR reader loading ...", t0)
reader = easyocr.Reader(["ja", "en"], gpu=False)
images = pdf2image.convert_from_path(PDF_PATH)
text = ""
for img in images:
    text += "\n".join(reader.readtext(np.array(img), detail=0)) + "\n"
log(f"OCR done ({len(text)} chars)", t0)

# チャンク（雑に300字ずつ）
chunks = [text[i:i+300] for i in range(0, len(text), 300)]
log(f"{len(chunks)} chunks", t0)

log("embedding start (bge-m3) ...", t0)
resp = ollama.embed(model=EMBED_MODEL, input=chunks)
embs = resp["embeddings"]
log(f"embedding done ({len(embs)} vectors, dim={len(embs[0])})", t0)

log("chroma write start ...", t0)
client = chromadb.PersistentClient(path="./.chroma_test")
col = client.get_or_create_collection(name="pipeline_test")
col.add(documents=chunks, ids=[f"t{i}" for i in range(len(chunks))], embeddings=embs)
log("chroma write done", t0)

log("=== ALL DONE ===", t0)
