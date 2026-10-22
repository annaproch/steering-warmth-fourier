from huggingface_hub import snapshot_download

if __name__ == '__main__':
    snapshot_download('NousResearch/Meta-Llama-3.1-8B-Instruct',
                      local_dir='conciseness/models/llama-3.1-8b-instruct',
                      allow_patterns=['*.json', '*.safetensors', 'tokenizer.model'],
                      max_workers=4)
