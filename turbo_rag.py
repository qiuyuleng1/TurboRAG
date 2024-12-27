import logging
import os
import sys
import torch
import json
import time
from tabulate import tabulate
import argparse
from qwen2_turbo import Qwen2TurboForCausalLM
from qwen2_block_attn import Qwen2BlockAttnForCausalLM
from transformers import AutoTokenizer

# Llama Index Related
from llama_index.core import Settings, load_index_from_storage, StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import BaseNode, TextNode, NodeWithScore

from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    repeat_kv,
    rotate_half,
    Qwen2DecoderLayer,
    Qwen2Model,
    Qwen2ForCausalLM,
    Qwen2Config,
    Cache
)


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

PREFIX = '''<|im_start|>system
You are an accurate and reliable AI assistant that can answer questions with the help of external documents. Please note that external documents may contain noisy information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate ’I can not answer the question because of the insufficient information in documents.‘.<|im_end|><|im_start|>user\nDocs:'''
# PREFIX = "Docs:"

def parse_json_to_nodes(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes_with_scores = []
    for chunk_key, chunk_data in data["docstore/data"].items():
        chunk_info = chunk_data["__data__"]
        
        # Create TextNode obj
        text_node = TextNode(
            id_=chunk_info["id_"],
            embedding=chunk_info["embedding"],
            metadata=chunk_info["metadata"],
            excluded_embed_metadata_keys=chunk_info["excluded_embed_metadata_keys"],
            excluded_llm_metadata_keys=chunk_info["excluded_llm_metadata_keys"],
            relationships=chunk_info["relationships"],
            text=chunk_info["text"],
            mimetype=chunk_info["mimetype"],
            start_char_idx=chunk_info["start_char_idx"],
            end_char_idx=chunk_info["end_char_idx"],
            text_template=chunk_info["text_template"],
            metadata_template=chunk_info["metadata_template"],
            metadata_seperator=chunk_info["metadata_seperator"],
        )
        
        # Create NodeWithScore obj, set score to 0
        node_with_score = NodeWithScore(node=text_node, score=0)
        nodes_with_scores.append(node_with_score)
    
    return nodes_with_scores


def stack_past_key_values(past_key_values_list):
    num_layers = len(past_key_values_list[0])
    batch_past_key_values = []
    for layer in range(num_layers):
        keys = torch.cat([past_key_values[layer][0] for past_key_values in past_key_values_list], dim=2)
        values = torch.cat([past_key_values[layer][1] for past_key_values in past_key_values_list], dim=2)
        batch_past_key_values.append((keys, values))
    return tuple(batch_past_key_values)

def qa_to_prompt(chunk_list, query):
    chunk_str = ".".join(chunk_list)

    chunk_str_ids = tokenizer.encode(PREFIX+"\n"+chunk_str, return_tensors='pt').to(model.device)
    print(f"chunk_str_ids = {len(chunk_str_ids[0])}")

    prompt = f'''{PREFIX}\n{chunk_str}\n\nQuestion: {query}<|im_end|><|im_start|>assistant\n'''
    # prompt = f'''{PREFIX}{chunk_str}{query}'''
    # prompt = f'''{chunk_str}{query}'''
    # print(prompt)
    return prompt

# Parse command-line arguments at global scope
parser = argparse.ArgumentParser(description='RAG with KV Cache Benchmarking Script')
parser.add_argument('--model_name', type=str, help='Path to the pretrained language model')
parser.add_argument('--embedding_model_name', type=str, help='Embedding model name or path')
parser.add_argument('--storage_dir', type=str, default='doc_emb', help='Directory where the index storage is located')
parser.add_argument('--query_file', type=str, default='./questions/query.jsonl', help='Path to the file containing queries')
parser.add_argument('--num_questions', type=int, default=50, help='Number of questions to process')
parser.add_argument('--similarity_top_k', type=int, default=4, help='Number of topk most relevant chunks')
parser.add_argument('--use_flash_attn', action='store_true', help='Use FlashAttention2')
parser.add_argument('--dtype', type=str, default='fp32', help='Data type for llm inference')
parser.set_defaults(use_chunk_cache=True)
args = parser.parse_args()

# Set up device globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model and tokenizer globally
attn_implementation = "flash_attention_2" if args.use_flash_attn else None
torch_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

# # model = Qwen2TurboForCausalLM.from_pretrained(
# # model = Qwen2ForCausalLM.from_pretrained(
# model = Qwen2BlockAttnForCausalLM.from_pretrained(
#     args.model_name,
#     torch_dtype=torch.bfloat16,
#     attn_implementation=attn_implementation).to(device)

if "_w_positions" in args.storage_dir:
    print("Use block attn.")
    model = Qwen2BlockAttnForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype
    ).to(device)
elif "_wo_positions" in args.storage_dir:
    print("Use turbo attn.")
    model = Qwen2TurboForCausalLM.from_pretrained(
        args.model_name, torch_dtype=torch_dtype
    ).to(device)
else:  # TODO
    model = Qwen2ForCausalLM.from_pretrained( 
        args.model_name, torch_dtype=torch_dtype
    ).to(device)
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

# Set up embedding model and index
Settings.embed_model = HuggingFaceEmbedding(model_name=args.embedding_model_name)
storage_context = StorageContext.from_defaults(persist_dir=args.storage_dir)
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=args.similarity_top_k)

inputs_prefix = tokenizer([PREFIX], return_tensors="pt",padding=True)
outputs_prefix = model(
    inputs_prefix['input_ids'].to(device), 
    attention_mask = inputs_prefix['attention_mask'].to(device), 
    use_cache=True
)
prefix_kvcache = outputs_prefix.past_key_values
# print(f"prefix_kvcache shape = {prefix_kvcache[0][0].shape}")  # (batch_size, num_heads, sequence_length, embed_size_per_head)

def load_kvcache(cache_file_path):
    return torch.load(cache_file_path, weights_only=True)

def query_with_kvcache(query_text, use_chunk_cache=True):
    query_bundle = QueryBundle(query_str=query_text)

    # print("Retriever type:", type(retriever))

    if os.environ.get("retrive") == "skip":
        print("skip retrive!")
        retrieved_nodes = parse_json_to_nodes(f"{args.storage_dir}/docstore.json")
    else:
        retrieved_nodes = retriever.retrieve(query_bundle)  # <class 'llama_index.core.indices.vector_store.retrievers.retriever.VectorIndexRetriever'>

    # print(f"len(retrieved_nodes) = {len(retrieved_nodes)}")
    # print(retrieved_nodes)

    kvcache_list = [prefix_kvcache]

    chunk_list = []
    prefix_ids = tokenizer.encode(PREFIX, return_tensors='pt').to(model.device)
    print(f"prefix_ids len={len(prefix_ids[0])}")
    chunk_token_count_list = [len(prefix_ids[0])]
    # chunk_token_count_list = []
    for node_with_score in retrieved_nodes:
    # for node_with_score in list(reversed(retrieved_nodes)):
        node = node_with_score.node  
        if use_chunk_cache:
            kvcache = torch.load(node.metadata["kvcache_file_path"], weights_only=True)
            # print(f"kvcache shape = {kvcache[0][0].shape}")
            kvcache_list.append(kvcache)
            chunk_token_count = tokenizer.encode(node.text, return_tensors='pt').to(model.device)
            chunk_token_count_list.append(len(chunk_token_count[0]))
            # chunk_token_count_list.append(stack_past_key_values([kvcache]).seen_tokens)
        chunk_list.append(node.text)
    print(f"chunk_token_count_list = {chunk_token_count_list}, chunk sum (with prefix) = {sum(chunk_token_count_list)}")

    query_ids = tokenizer.encode(query_text, return_tensors='pt').to(model.device)
    print(f"query_ids = {len(query_ids[0])}")
    if use_chunk_cache:
        os.environ["CHUNK_TOKEN_COUNT_LIST"] = json.dumps(chunk_token_count_list)
    prompt = qa_to_prompt(chunk_list, query_text)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    print(f"input_ids = {len(input_ids[0])}")
    past_kvcache = stack_past_key_values(kvcache_list) if use_chunk_cache else None
    print(f"type past_kvcache = {type(past_kvcache)}")
    eos_token_ids = [151645,151643]
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        past_key_values=past_kvcache,
        pad_token_id=tokenizer.eos_token_id,
        # do_sample=True,
        do_sample=False,    # 我不想要随机性
        eos_token_id=eos_token_ids,
        # temperature = 0.1,
        # top_p = 0.9,
    )

    generated_ids = [
        output_ids[len(input_id):] for input_id, output_ids in zip(input_ids, outputs)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"query:{query_text}\noutputs:{response}")

    # print(outputs)
    return response

if __name__ == "__main__":
    questions = []
    with open(args.query_file) as file:
        for item in file:
            data = json.loads(item)
            questions.append(data["query"])
    questions = questions[:args.num_questions]

    # # Test the average time taken for RAG with document composite_positions KV Cache
    # os.environ["USE_CHUNK_CACHE"] = "composite_positions"
    # print(f'USE_CHUNK_CACHE={os.environ["USE_CHUNK_CACHE"]}')
    # results_with_composite_kvcache = {}
    # start = time.perf_counter()
    # for query in questions:
    #     results_with_composite_kvcache[query] = query_with_kvcache(query)
    # end = time.perf_counter()
    # use_time = end - start
    # avg_time_with_composite_positions_cache = use_time / len(questions)

    # Test the average time taken for RAG with document reordered_positions KV Cache
    os.environ["USE_CHUNK_CACHE"] = "reordered_positions"
    print(f'USE_CHUNK_CACHE={os.environ["USE_CHUNK_CACHE"]}')
    results_with_reordered_kvcache = {}
    start = time.perf_counter()
    for query in questions:
        results_with_reordered_kvcache[query] = query_with_kvcache(query)
    end = time.perf_counter()
    use_time = end - start
    avg_time_with_reordered_positions_cache = use_time / len(questions)
    
    # Test the average time taken for RAG without document chunk KV Cache
    os.environ["USE_CHUNK_CACHE"] = "false"
    print(f'USE_CHUNK_CACHE={os.environ["USE_CHUNK_CACHE"]}')
    results_without_kvcache = {}
    start = time.perf_counter()
    for query in questions:
        results_without_kvcache[query] = query_with_kvcache(query, use_chunk_cache=False)
    end = time.perf_counter()
    use_time_without_cache = end - start
    avg_time_without_cache = use_time_without_cache / len(questions)

    # # Test loss of reverse RoPE
    # os.environ["USE_CHUNK_CACHE"] = "test_reverse_RoPE"
    # print(f'USE_CHUNK_CACHE={os.environ["USE_CHUNK_CACHE"]}')
    # results_without_test_kvcache = {}
    # start = time.perf_counter()
    # for query in questions:
    #     results_without_test_kvcache[query] = query_with_kvcache(query, use_chunk_cache=False)
    # end = time.perf_counter()
    # use_time_without_cache = end - start
    # avg_time_without_cache_test_reverse_RoPE = use_time_without_cache / len(questions)

    # # Prepare data for tabular output
    # results = [
    #     ["With Composite Positions KV Cache", f"{avg_time_with_composite_positions_cache:.6f} seconds"],
    #     ["With Reordered Positions KV Cache", f"{avg_time_with_reordered_positions_cache:.6f} seconds"],
    #     ["Without KV Cache", f"{avg_time_without_cache:.6f} seconds"],
    #     ["Without KV Cache test reverse RoPE", f"{avg_time_without_cache_test_reverse_RoPE:.6f} seconds"]
    # ]

    # # Print the results in a table format
    # print(tabulate(results, headers=["Method", "Average Time"], tablefmt="grid"))

    # for query in results_with_composite_kvcache.keys():
    #     print(f"{query}\nWith composite Cache:{results_with_composite_kvcache[query]}\n With reordered Cache:{results_with_reordered_kvcache[query]}\n Without Turbo Cache:{results_without_kvcache[query]}\n Without test KV Cache:{results_without_test_kvcache[query]}\n \n")
