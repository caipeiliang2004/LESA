
Prepare enviroment
```bash
# create environment 
conda create -n qwen python=3.10
conda activate qwen
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.55.4 peft
pip install git+https://github.com/huggingface/diffusers

# evaluate
pip install opencv-python lpips scikit-image image-reward
pip install git+https://github.com/openai/CLIP.git
pip install calflops
```

Set environment variables (in `.bashrc` file)
```bash
export XDG_CACHE_HOME="/path/to/.cache"
export HF_ENDPOINT="https://hf-mirror.com"
```

Download models and dataset
```bash
hf download Qwen/Qwen-Image
hf download lightx2v/Qwen-Image-Lightning

# evaluate
hf download zai-org/ImageReward
hf download laion/CLIP-ViT-g-14-laion2B-s12B-b42K
```

Change model path
```python
# train.py line 65 91 97 sample.py line 59 85 91
"Qwen/Qwen-Image",
"Qwen/Qwen-Image",
"lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V2.0.safetensors"

# evaluate.py line 175 176
parser.add_argument('--clip_model_path', type=str, default="laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
parser.add_argument('--imagereward_model_path', type=str, default="zai-org/ImageReward")
```