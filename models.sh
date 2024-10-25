#!/bin/bash

# Create a directory for models if it doesn't exist
mkdir -p models

# Clone the MiniLM model repository
git clone https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 models/all-MiniLM-L6-v2

# Download the Llama-2 model files to the models directory
wget -O models/llama-2-7b-chat.ggmlv3.q4_1.bin https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_1.bin

wget -O models/llama-2-13b-german-assistant-v2.ggmlv3.q4_0.bin https://huggingface.co/TheBloke/llama-2-13B-German-Assistant-v2-GGML/resolve/main/llama-2-13b-german-assistant-v2.ggmlv3.q4_0.bin
