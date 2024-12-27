export http_proxy=http://proxy.ims.intel.com:911
export https_proxy=http://proxy.ims.intel.com:911

# cases=("case1" "case2" "case3" "case4" "case5" "case6" "case7")
# cases=("case11" "case12" "case13" "case14" "case15" "case16" "case17")
# cases=("case22" "case23" "case24")
cases=("case32" "case33" "case34")

# # turbo attn
# for case in "${cases[@]}"; do
#     documents_dir="count_star_documents_${case}"
#     kv_cache_storage_dir="count_star_${case}_chunk_kvcache_wo_positions"
#     index_storage_dir="count_star_${case}_doc_emb_wo_positions"
#     python chunk_cache.py --documents_dir "$documents_dir" --kv_cache_storage_dir "$kv_cache_storage_dir" --index_storage_dir "$index_storage_dir"
# done

# block attn
for case in "${cases[@]}"; do
    documents_dir="count_star_documents_${case}"
    kv_cache_storage_dir="count_star_${case}_chunk_kvcache_w_positions"
    index_storage_dir="count_star_${case}_doc_emb_w_positions"
    python chunk_cache.py --documents_dir "$documents_dir" --kv_cache_storage_dir "$kv_cache_storage_dir" --index_storage_dir "$index_storage_dir"
done