import torch
from transformers import AutoTokenizer
from qwen2_turbo import Qwen2TurboForCausalLM
from qwen2_block_attn import Qwen2BlockAttnForCausalLM
from typing import List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import os

# LlamaIndex related
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    PromptHelper,
    Document,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.indices.query.schema import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import SimpleVectorStore

import argparse
parser = argparse.ArgumentParser(description='KV Cache Preprocessing Script')
parser.add_argument('--documents_dir', type=str, help='Directory containing documents to be processed')
parser.add_argument('--kv_cache_storage_dir', type=str, help='Directory where the chunk kv cache storage is located')
parser.add_argument('--index_storage_dir', type=str, help='Directory where the llama vector store index storage is located')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_name = "/media/model-space/Qwen2.5-7B-Instruct" 
# model_name = "/media/model-space/Qwen/Qwen2.5-32B-Instruct"

if "_w_positions" in args.kv_cache_storage_dir and "_w_positions" in args.index_storage_dir:
    print("Use block attn.")
    model = Qwen2BlockAttnForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
elif "_wo_positions" in args.kv_cache_storage_dir and "_wo_positions" in args.index_storage_dir:
    print("Use turbo attn.")
    model = Qwen2TurboForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
else:
    raise ValueError(
        "Both args.kv_cache_storage_dir and args.index_storage_dir must \
            contain either '_w_positions' or '_wo_positions'."
    )
tokenizer = AutoTokenizer.from_pretrained(model_name)

splitter = TokenTextSplitter(
    tokenizer=tokenizer.encode,
    chunk_size=512,
    chunk_overlap=10
)

output_path = args.kv_cache_storage_dir
if not os.path.exists(output_path):
    os.makedirs(output_path)

def process_chunk(chunk_text, chunk_id):
    chunk_text = "<|doc_start|>" + chunk_text + "<|doc_end|>"
    inputs = tokenizer(chunk_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, use_cache=True)
    
    past_key_values = outputs.past_key_values
    
    kvcache_file_path = f'{output_path}/kvcache_chunk_{chunk_id}.pt'
    torch.save(past_key_values, kvcache_file_path)
    
    node = TextNode(
        text=chunk_text,
        id_=f"chunk_{chunk_id}",
        metadata={
            "kvcache_file_path": kvcache_file_path
        }
    )
    
    return node

class KVCachedNodeParser(SimpleNodeParser):
    def get_nodes_from_documents(
        self,
        documents: List[Document],
        **kwargs,
    ) -> List[BaseNode]:
        nodes = []
        for doc_id, document in tqdm(enumerate(documents)):
            doc_text = document.get_content()
            chunk_texts = splitter.split_text(doc_text)
            
            for chunk_id, chunk_text in enumerate(chunk_texts):
                node = process_chunk(chunk_text, f"{doc_id}_{chunk_id}")
                nodes.append(node)
        return nodes

embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
)

vector_store = SimpleVectorStore()
node_parser = KVCachedNodeParser()
documents = SimpleDirectoryReader(args.documents_dir).load_data()
nodes = node_parser.get_nodes_from_documents(documents)

index = VectorStoreIndex(
    nodes=nodes,
    embed_model=embed_model,
    vector_store=vector_store,
)

index.storage_context.persist(persist_dir=args.index_storage_dir)