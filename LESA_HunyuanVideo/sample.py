import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args, parse_args_predictor
from hyvideo.inference import HunyuanVideoSampler


def main():
    args = parse_args()
    opts = parse_args_predictor()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

    # Load predictor
    from predictor import Predictor, Config
    predictor = Predictor(Config(), num_steps=args.infer_steps, enable_training=False)

    weights_path = f"{opts.weights_dir}/predictor_hunyuanvideo_N{opts.interval}_E{opts.first_enhance}.pt"
    assert os.path.exists(weights_path), f"Weights file {weights_path} does not exist"
    predictor.load_model(weights_path)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    kwargs={
        'num_steps': args.infer_steps, 
        'height': args.video_size[0],
        'width': args.video_size[1],
        'test_FLOPs': opts.test_FLOPs,
        'monitor_gpu_usage': opts.monitor_gpu_usage,
        'data_prepare': opts.data_prepare,
        'data_dir': opts.data_dir,
        'phase': opts.phase,
        'idx': -1,
        'interval': opts.interval,
        'first_enhance': opts.first_enhance,
    }

    from predictor import cache_init
    cache_dic, current = cache_init(**kwargs)

    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        height=args.video_size[0],
        width=args.video_size[1],
        video_length=args.video_length,
        seed=args.seed,
        negative_prompt=args.neg_prompt,
        infer_steps=args.infer_steps,
        guidance_scale=args.cfg_scale,
        num_videos_per_prompt=args.num_videos,
        flow_shift=args.flow_shift,
        batch_size=args.batch_size,
        embedded_guidance_scale=args.embedded_cfg_scale,
        cache_dic=cache_dic,
        current=current,
        predictor=predictor,
    )
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python sample_video.py --video-size 480 640 --video-length 65 --prompt "A cat walks on the grass, realistic style." --flow-reverse --save-path samples/test
    # --interval 7 --first_enhance 3 --test_FLOPs --monitor_gpu_usage