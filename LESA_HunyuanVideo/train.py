#!/usr/bin/env python3
import os
from tqdm import tqdm
# First, set environment variables to disable tokenizers warning messages
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import json
import contextlib
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args, parse_args_predictor
from hyvideo.inference import HunyuanVideoSampler

@contextlib.contextmanager
def suppress_output():
    """
    Temporarily suppress standard output, standard error, and loguru output.
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        with open(os.devnull, "w") as devnull:
            sys.stdout = devnull
            sys.stderr = devnull
            # Disable loguru output
            logger.disable("")
            yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        logger.enable("")

def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def main():
    # Get command-line arguments
    args = parse_args()
    opts = parse_args_predictor()

    prompts = read_prompts(opts.prompt_file)
    
    # Load model
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` does not exist: {models_root_path}")

    # Load predictor
    from predictor import Predictor, Config
    predictor = Predictor(Config(), num_steps=args.infer_steps, enable_training=True)

    os.makedirs(f"{opts.weights_dir}", exist_ok=True)
    weights_path = f"{opts.weights_dir}/predictor_hunyuanvideo_N{opts.interval}_E{opts.first_enhance}.pt"
    if os.path.exists(weights_path):
        print("Loading predictor...")
        predictor.load_model(weights_path)
    
    # Create save directory
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)
    
    # Load the sampler (only load the model once)
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    # Update sampler internal parameters
    args = hunyuan_video_sampler.args

    progress_bar = tqdm(total=len(prompts) * opts.iterations, desc="Generating videos")
    idx = 0
    
    for _ in range(len(prompts) * opts.iterations):
        cur_save_path = f"{save_path}/video_{idx}.mp4"
        
        kwargs={
            'num_steps': args.infer_steps, 
            'height': args.video_size[0],
            'width': args.video_size[1],
            'test_FLOPs': opts.test_FLOPs,
            'monitor_gpu_usage': opts.monitor_gpu_usage,
            'data_prepare': opts.data_prepare,
            'data_dir': opts.data_dir,
            'phase': opts.phase,
            'idx': idx,
            'interval': opts.interval,
            'first_enhance': opts.first_enhance,
        }

        from predictor import cache_init
        cache_dic, current = cache_init(**kwargs)

        with suppress_output():
            outputs = hunyuan_video_sampler.predict(
                prompt=prompts[idx],
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=1,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale,
                cache_dic=cache_dic,
                current=current,
                predictor=predictor,
            )
        samples = outputs['samples']
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f"Sample saved to: {cur_save_path}")

        predictor.save_model(f"{opts.weights_dir}/predictor_hunyuanvideo_N{opts.interval}_E{opts.first_enhance}.pt")

        progress_bar.update(1)

        if idx == len(prompts) - 1:
            idx = 0
            continue

        idx += 1


if __name__ == "__main__":
    main()
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python train.py --video-size 480 640 --video-length 65 --flow-reverse --interval 7 --first_enhance 3 --save-path samples/test
