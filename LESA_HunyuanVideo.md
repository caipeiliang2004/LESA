
Prepare environment
```bash
# create environment
conda create --name hunyuanvideo python==3.10.9
conda activate hunyuanvideo
# install torch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python==4.9.0.80 diffusers==0.31.0 transformers==4.46.3 tokenizers==0.20.3 accelerate==1.1.1 pandas==2.0.3 numpy==1.24.4 einops==0.7.0 tqdm==4.66.2 loguru==0.7.2 imageio==2.34.0 imageio-ffmpeg==0.5.1 safetensors==0.4.3 gradio==5.0.0 ninja
# install flash-attn
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
rm flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# install xfuser(pin dependency version)
pip install xfuser==0.4.0 torch==2.4.0 accelerate==1.1.1 transformers==4.46.3 yunchang==0.5.0 opencv-python==4.9.0.80 imageio==2.34.0 imageio-ffmpeg==0.5.1 flash_attn==2.6.3 diffusers==0.31.0

# evaluate
pip install calflops
```

Download model
Follow the official guide

Set ckpt path
```python
# hyvideo/constants.py line 67
MODEL_BASE = os.getenv("MODEL_BASE", "./ckpts")
# hyvideo/config.py line 246
default="ckpts",
# hyvideo/config.py line 252
default="ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
```
