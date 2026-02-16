# mistral-vllm-gcp
Deploy and serve the Mistral-7B language model using vLLM on an NVIDIA T4 GPU with only 16GB VRAM. 

## About

- **Quantized Model Deployment**: Uses AWQ quantization to fit Mistral-7B in 16GB VRAM
- **OpenAI-Compatible API**: compatible with openai APIs
- **PoC ONLY**: Not ready for PROD. useful for testing purposes only
- **Memory Efficient**: Only uses ~4GB VRAM for the model, leaving room for concurrent requests. I could have used even a smaller instance :D 

## Prerequisites
- NVIDIA T4 GPU (16GB VRAM)
- Ubuntu 22.04 or later
- CUDA Drivers 12.1+
- Python 3.10+
- 100GB+ disk space 

## Installation

### 1. Set Up GCP VM (Optional)

Deployed on GCP us-central-1.. add your VPC and subnet names

```bash
gcloud compute instances create llm-gpu-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-standard \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True"
```

### 2. Verify CUDA Installation

```bash
# Check NVIDIA driver
nvidia-smi

# Verify CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# I had to install CUDA drivers manually
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common

sudo ubuntu-drivers autoinstall

sudo reboot
```

### 3. Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
pip install vllm

# Install additional dependencies
pip install requests
```

## 🎯 Quick Start

### Start the vLLM Server

```bash
#I had to explicitly set environment variable to use stable engine
export VLLM_USE_V1=0

# Start the server
python -m vllm.entrypoints.openai.api_server \
    --model TheBloke/Mistral-7B-Instruct-v0.2-AWQ \
    --quantization awq \
    --dtype half \
    --attention-backend TORCH_SDPA \
    --host 0.0.0.0 \
    --port 8000
```

### Test the API

```bash
# Check available models
curl http://localhost:8000/v1/models

# Send a completion request
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TheBloke/Mistral-7B-Instruct-v0.2-AWQ",
    "prompt": "tell me about the history of artificial intellignece:",
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

## Configuration

### Memory Settings

Adjust based on your GPU:

```bash
--gpu-memory-utilization 0.75 \
--max-model-len 2048

```

### Attention Backends

Available options (i went with TORCH_SDPA):

```bash
# TORCH_SDPA (recommended, no compilation needed)
--attention-backend TORCH_SDPA
```


## Performance

On NVIDIA T4:

- **Model Size**: 3.88 GB VRAM
- **KV Cache**: 7.61 GB available
- **Throughput**: ~15-20 tokens/second 
- **Max Concurrent Requests**: ~15 (with max 4096 context length)
