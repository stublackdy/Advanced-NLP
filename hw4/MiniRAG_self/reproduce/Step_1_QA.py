# from huggingface_hub import login
# your_token = "INPUT YOUR TOKEN HERE"
# login(your_token)

import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gc

import csv
from tqdm import trange
from minirag import MiniRAG, QueryParam
from minirag.llm import (
    hf_model_complete,
    hf_embed,
)
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

import argparse


def get_args():
    parser = argparse.ArgumentParser(description="MiniRAG")
    parser.add_argument("--model", type=str, default="qwen")
    parser.add_argument("--outputpath", type=str, default="/root/akshat/MiniRAG_self/logs/14_weeks.csv")
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
rag = MiniRAG(
    working_dir=WORKING_DIR,
    llm_model_func=hf_model_complete,
    # llm_model_func=gpt_4o_mini_complete,
    llm_model_max_token_size=200,
    llm_model_name=LLM_MODEL,
    embedding_func=EmbeddingFunc(
        embedding_dim=384,
        max_token_size=1000,
        func=lambda texts: hf_embed(
            texts,
            tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
            embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL).to(device),
        ),
    ),
)

# Now QA
QUESTION_LIST = []
GA_LIST = []
with open(QUERY_PATH, mode="r", encoding="utf-8") as question_file:
    reader = csv.DictReader(question_file)
    for row in reader:
        QUESTION_LIST.append(row["Question"])
        GA_LIST.append(row["Gold Answer"])




# def run_experiment(output_path):
#     headers = ["Question", "Gold Answer", "query", "context_json_data",  "sys_prompt", "minirag"]

#     q_already = []
#     if os.path.exists(output_path):
#         with open(output_path, mode="r", encoding="utf-8") as question_file:
#             reader = csv.DictReader(question_file)
#             for row in reader:
#                 q_already.append(row["Question"])

#     row_count = len(q_already)
#     print("row_count", row_count)
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     with open(output_path, mode="a", newline="", encoding="utf-8") as log_file:
#         writer = csv.writer(log_file)

    
#         if row_count == 0:
#             writer.writerow(headers)

#         for QUESTIONid in range(len(QUESTION_LIST))[4:]:  #
#             QUESTION = QUESTION_LIST[QUESTIONid]
#             Gold_Answer = GA_LIST[QUESTIONid]
#             print()
#             print("QUESTION", QUESTION)
#             print("Gold_Answer", Gold_Answer)

#             # try:
#             query, context_json_data,  sys_prompt, minirag_answer = (
#                 rag.query(QUESTION, param=QueryParam(mode="mini"))
#                 # .replace("\n", "")
#                 # .replace("\r", "")
#             )


#                 # print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n")
#                 # print(f"question: {QUESTION}")
#                 # print(f"Context: {minirag_answer}")
#                 # print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n\n")
#             # except Exception as e:
#             #     print("Error in minirag_answer", e)
#             #     minirag_answer = e

#             writer.writerow([QUESTION, Gold_Answer, query, context_json_data,  sys_prompt, minirag_answer])
#             # break
#             # torch.cuda.empty_cache()
#             # torch.cuda.ipc_collect()
#             # gc.collect()

#     print(f"Experiment data has been recorded in the file: {output_path}")

import os
import pandas as pd


def run_experiment(input_csv_path, output_csv_path):
    # Read input questions
    df_in = pd.read_csv(input_csv_path)
    print(f"Loaded {len(df_in)} questions from {input_csv_path}")

    # Load already processed results if CSV exists
    if os.path.exists(output_csv_path):
        df_out = pd.read_csv(output_csv_path)
        processed = set(df_out["Question"].tolist())
        print(f"Found {len(processed)} already processed questions")
    else:
        df_out = pd.DataFrame(columns=[
            "Question",
            "Gold Answer",
            "query",
            "context_json_data",
            "sys_prompt",
            "minirag"
        ])
        processed = set()

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

    # Process only unprocessed questions
    for i, row in df_in.iterrows():
        if i<4:
            continue


        QUESTION = row["Question"]
        Gold_Answer = row["Gold Answer"]

        # if QUESTION in processed:
        #     print(f"Skipping (already done): {QUESTION}")
        #     continue

        print(f"\nProcessing {i+1}/{len(df_in)}")
        print("QUESTION:", QUESTION)

        try:
            query, context_json_data, sys_prompt, minirag_answer = rag.query(
                QUESTION, param=QueryParam(mode="mini")
            )
        except Exception as e:
            print("Error in query:", e)
            query, context_json_data, sys_prompt, minirag_answer = "", "", "", str(e)

        # Append result as a new row
        new_row = pd.DataFrame([{
            "Question": QUESTION,
            "Gold Answer": Gold_Answer,
            "query": query,
            "context_json_data": context_json_data,
            "sys_prompt": sys_prompt,
            "minirag": minirag_answer
        }])
        new_row.to_csv(output_csv_path, mode="a", header=not os.path.exists(output_csv_path), index=False)
        print(f"Appended → {output_csv_path}")

        # Optional cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    print(f"\n✅ All query results recorded in: {output_csv_path}")


# if __name__ == "__main__":

run_experiment(QUERY_PATH, OUTPUT_PATH)
