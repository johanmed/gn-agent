# Pass models through HuggingFace ids

CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --port 7501 --model-path Tsunami-th/Tsunami-0.5-7B-Instruct

CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server --port 7502 --model-path microsoft/Phi-3-mini-4k-instruct

