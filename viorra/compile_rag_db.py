import os
import glob
import pickle
import numpy as np
import turbovec
from fastembed import TextEmbedding

# Paths
DATASET_DIR = r"C:\Users\Sardor\.gemini\antigravity\scratch\viorra_dataset"
USER_DATA_DIR = r"C:\Users\Sardor\AppData\Local\Viorra"

paired_dir = os.path.join(DATASET_DIR, "paired_data")
unpaired_dir = os.path.join(DATASET_DIR, "unpaired_data")

corpus_texts = []
corpus_feedback = []

# 1. Load paired data
print("Scanning paired data...")
categories = ["us_commonapp", "ucas_uk", "ucla_piq", "mba_insead"]
for cat in categories:
    cat_dir = os.path.join(paired_dir, cat)
    if not os.path.exists(cat_dir):
        continue
    essay_files = glob.glob(os.path.join(cat_dir, "*_essay.md"))
    for essay_file in essay_files:
        basename = os.path.basename(essay_file)
        prefix = basename.split("_essay.md")[0]
        feedback_file = os.path.join(cat_dir, f"{prefix}_essay_feedback.md")
        
        try:
            with open(essay_file, "r", encoding="utf-8") as f:
                essay_content = f.read().strip()
            feedback_content = ""
            if os.path.exists(feedback_file):
                with open(feedback_file, "r", encoding="utf-8") as f:
                    feedback_content = f.read().strip()
            
            if essay_content:
                corpus_texts.append(essay_content)
                corpus_feedback.append(feedback_content)
        except Exception as e:
            print(f"Error reading {essay_file}: {e}")

# 2. Load unpaired data
print("Scanning unpaired data...")
for cat in categories:
    cat_dir = os.path.join(unpaired_dir, cat)
    if not os.path.exists(cat_dir):
        continue
    essay_files = glob.glob(os.path.join(cat_dir, "*_essay.md"))
    for essay_file in essay_files:
        try:
            with open(essay_file, "r", encoding="utf-8") as f:
                essay_content = f.read().strip()
            if essay_content:
                corpus_texts.append(essay_content)
                corpus_feedback.append("No historical feedback available. Refer to the essay for style/tone reference.")
        except Exception as e:
            print(f"Error reading {essay_file}: {e}")

print(f"Total documents loaded: {len(corpus_texts)}")

if len(corpus_texts) == 0:
    print("No documents found to index!")
    exit(1)

# 3. Generate embeddings
print("Initializing FastEmbed...")
embedder = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2", threads=1)

print("Generating embeddings (this may take a minute)...")
embeddings = list(embedder.embed(corpus_texts))
embeddings_np = np.array(embeddings, dtype=np.float32)

print(f"Embeddings shape: {embeddings_np.shape}")

# 4. Compile TurboVec index
print("Compiling TurboVec Index...")
dim = embeddings_np.shape[1]
index = turbovec.TurboQuantIndex(dim=dim, bit_width=4)
index.add(embeddings_np)
index.prepare()

# 5. Save outputs
os.makedirs(USER_DATA_DIR, exist_ok=True)
index_file = os.path.join(USER_DATA_DIR, "viorra_index_local.tv")
corpus_file = os.path.join(USER_DATA_DIR, "viorra_corpus_local.pkl")

index.write(index_file)
with open(corpus_file, "wb") as f:
    pickle.dump((corpus_texts, corpus_feedback), f)

print(f"Success! TurboVec index saved to {index_file}")
print(f"Corpus saved to {corpus_file}")
