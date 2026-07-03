import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import time
import numpy as np
import easyocr
import pdf2image

PDF_PATH = "../data/travel_rule.pdf"

print("=== start ===", flush=True)
t0 = time.time()

print("[1] loading OCR reader ...", flush=True)
reader = easyocr.Reader(["ja", "en"], gpu=False)
print(f"    reader ready ({time.time()-t0:.1f}s)", flush=True)

print("[2] converting PDF to images ...", flush=True)
images = pdf2image.convert_from_path(PDF_PATH)
print(f"    {len(images)} pages ({time.time()-t0:.1f}s)", flush=True)

text = ""
for i, img in enumerate(images):
    ts = time.time()
    print(f"[3] OCR page {i+1}/{len(images)} ...", flush=True)
    result = reader.readtext(np.array(img), detail=0)
    text += "\n".join(result) + "\n"
    print(f"    page {i+1} done ({time.time()-ts:.1f}s)", flush=True)

with open("ocr_out.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"=== done total {time.time()-t0:.1f}s -> ocr_out.txt ===", flush=True)
print("---- preview ----", flush=True)
print(text[:500], flush=True)
