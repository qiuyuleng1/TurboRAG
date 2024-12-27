export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911

export retrive=skip
# unset retrive

# dtype=bf16
# case_number=7
# w_or_wo_positions=wo
# logs=logs/count_star_case${case_number}_${w_or_wo_positions}_positions_$dtype.log
# # logs=logs/tmp.log

# python turbo_rag.py --model_name "/media/model-space/Qwen2.5-7B-Instruct" --dtype $dtype --embedding_model_name="BAAI/bge-small-en-v1.5" --storage_dir count_star_case${case_number}_doc_emb_${w_or_wo_positions}_positions --query_file count_star_questions/query.jsonl --num_questions 4 --similarity_top_k 10 2>&1 | tee $logs


#!/bin/bash

dtype=bf16
# case_numbers=(11 12 13 14 15 16 17)
case_numbers=(32 33 34)
# case_numbers=(22)
# w_or_wo_positions=("w" "wo")  # block attn, turbo attn
w_or_wo_positions=("w")


for case_number in "${case_numbers[@]}"; do
    for w_or_wo in "${w_or_wo_positions[@]}"; do
        logs="logs/count_star_case${case_number}_query4_${w_or_wo}_positions_$dtype.log"
        # logs="logs/tmp.log"

        echo $logs
        
        python turbo_rag.py \
            --model_name "/media/model-space/Qwen2.5-7B-Instruct" \
            --dtype "$dtype" \
            --embedding_model_name="BAAI/bge-small-en-v1.5" \
            --storage_dir "count_star_case${case_number}_doc_emb_${w_or_wo}_positions" \
            --query_file "count_star_questions/query4.jsonl" \
            --num_questions 4 \
            --similarity_top_k 10 2>&1 | tee "$logs"
    done
done