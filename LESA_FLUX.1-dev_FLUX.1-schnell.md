
Prepare enviroment
```bash
# create environment 
conda create -n flux python=3.10
conda activate flux
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install .
pip install transformers==4.55.4

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
hf download --token YOUR_HFTOKEN black-forest-labs/FLUX.1-schnell
hf download --token YOUR_HFTOKEN black-forest-labs/FLUX.1-dev

hf download google/t5-v1_1-xxl
hf download openai/clip-vit-large-patch14

# evaluate
hf download zai-org/ImageReward
hf download laion/CLIP-ViT-g-14-laion2B-s12B-b42K
```

Change model path
```python
# src/flux/util.py line 301 302
repo_flow="flux1-dev.safetensors",
repo_ae="ae.safetensors",
# src/flux/util.py line 363 364
repo_flow="flux1-schnell.safetensors",
repo_ae="ae.safetensors",
# src/flux/util.py line 691
return HFEmbedder("google/t5-v1_1-xxl", max_length=max_length, torch_dtype=torch.bfloat16).to(device)
# src/flux/util.py line 695
return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, torch_dtype=torch.bfloat16).to(device)

# evaluate.py line 175 176
parser.add_argument('--clip_model_path', type=str, default="laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
parser.add_argument('--imagereward_model_path', type=str, default="zai-org/ImageReward")
```