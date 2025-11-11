# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
import time
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# TODO: add your Gemini API key
os.environ["GEMINI_API_KEY"] = ""

from minirag import MiniRAG
from minirag.llm import (
    gemini_25_flash_lite_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="gemini")
    parser.add_argument("--outputpath", type=str, default="./logs/Default_output.csv")
    parser.add_argument("--workingdir", type=str, default="./LiHua-World")
    parser.add_argument("--datapath", type=str, default="./dataset/LiHua-World/data/")
    parser.add_argument(
        "--querypath", type=str, default="./dataset/LiHua-World/qa/query_set.csv"
    )
    args = parser.parse_args()
    return args


args = get_args()

if args.model == "PHI":
    LLM_MODEL = "microsoft/Phi-3.5-mini-instruct"
elif args.model == "GLM":
    LLM_MODEL = "THUDM/glm-edge-1.5b-chat"
elif args.model == "MiniCPM":
    LLM_MODEL = "openbmb/MiniCPM3-4B"
elif args.model == "gemini":
    LLM_MODEL = "gemini-2.5-flash-lite"
elif args.model == "qwen":
    LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
else:
    print("Invalid model name")
    exit(1)

WORKING_DIR = args.workingdir
DATA_PATH = args.datapath
QUERY_PATH = args.querypath
OUTPUT_PATH = args.outputpath

print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=gemini_25_flash_lite_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
        ),
    ),
)


def find_txt_files(root_path):
    txt_files = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if file.endswith(".txt"):
                txt_files.append(os.path.join(root, file))
    return txt_files


WEEK_LIST = find_txt_files(DATA_PATH)

# Load progress file to skip already processed files
progress_file = os.path.join(WORKING_DIR, "processing_progress.txt")
processed_files = set()
if os.path.exists(progress_file):
    with open(progress_file, 'r', encoding='utf-8') as f:
        processed_files = set(line.strip() for line in f if line.strip())
    print(f"Resuming from previous run. Already processed: {len(processed_files)} files")

# Statistics counters
stats = {
    'total': len(WEEK_LIST),
    'already_processed': len(processed_files),
    'newly_processed': 0,
    'skipped_errors': 0,
    'failed_after_retries': 0
}

for WEEK in WEEK_LIST:
    id = WEEK_LIST.index(WEEK)
    print(f"\n{'='*60}")
    print(f"Progress: {id+1}/{len(WEEK_LIST)} | File: {os.path.basename(WEEK)}")
    print(f"{'='*60}")

    # Skip if already processed
    if WEEK in processed_files:
        print(f"✓ Already processed in previous run")
        continue

    max_retries = 5
    retry_count = 0
    success = False

    while not success and retry_count < max_retries:
        try:
            with open(WEEK, encoding='utf-8') as f:
                rag.insert(f.read())
            success = True
            # Save progress
            with open(progress_file, 'a', encoding='utf-8') as pf:
                pf.write(f"{WEEK}\n")
            stats['newly_processed'] += 1
            print(f"✓ Successfully processed!")
            time.sleep(2)  # Delay between files
        except Exception as e:
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ["rate_limit", "429", "timeout", "timed out", "connection", "network"]):
                retry_count += 1
                wait_time = 30 * retry_count
                print(f"Retryable error encountered: {type(e).__name__}")
                print(f"Waiting {wait_time} seconds before retry {retry_count}/{max_retries}...")
                time.sleep(wait_time)
            else:
                print(f"✗ Non-retryable error: {type(e).__name__} - {e}")
                print(f"Skipping this file and continuing...")
                stats['skipped_errors'] += 1
                break

    if not success:
        print(f"✗ Failed after {max_retries} retries. Skipping...")
        stats['failed_after_retries'] += 1
        continue

# Print final statistics
print(f"\n{'='*60}")
print("PROCESSING COMPLETE!")
print(f"{'='*60}")
print(f"Total files: {stats['total']}")
print(f"Already processed (from previous run): {stats['already_processed']}")
print(f"Newly processed: {stats['newly_processed']}")
print(f"Skipped due to non-retryable errors: {stats['skipped_errors']}")
print(f"Failed after retries: {stats['failed_after_retries']}")
print(f"Total successfully processed: {stats['already_processed'] + stats['newly_processed']}")
print(f"{'='*60}")