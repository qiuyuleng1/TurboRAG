export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911

python turbo_rag.py --model_name "/media/model-space/Qwen2.5-7B-Instruct" --embedding_model_name="BAAI/bge-small-en-v1.5" --storage_dir doc_emb_w_position --query_file questions_old/query.jsonl --num_questions 1 --similarity_top_k 3 2>&1 | tee output.log