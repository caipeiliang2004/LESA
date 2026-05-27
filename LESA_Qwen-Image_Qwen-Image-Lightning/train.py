import os
import math
import torch
from PIL import Image, ExifTags
from tqdm import tqdm
from dataclasses import dataclass
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler 
import torch.distributed as dist

from pipelines.qwenimage.pipeline_qwenimage import QwenImagePipeline


@dataclass
class SamplingOptions:
    prompts: list[str]          # List of prompts
    negative_prompt: str        # Negative prompt for guidance
    width: int                  # Image width
    height: int                 # Image height
    num_steps: int              # Number of sampling steps
    guidance_scale: float       # Guidance scale
    seed: int                   # Random seed
    model_name: str             # Model name
    output_dir: str             # Output directory
    test_FLOPs: bool            # Whether in FLOPs test mode
    monitor_gpu_usage: bool     # Whether to monitor GPU memory usage
    data_prepare: bool           # Whether to prepare data
    data_dir: str                # Data directory
    weights_dir: str             # Weights directory
    phase: str                   # Phase
    iterations: int              # Number of iterations
    interval: int                # Interval
    first_enhance: int           # First enhance


def main(opts: SamplingOptions):
    # Initialize distributed environment
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)

    # Task allocation for distributed processing
    total_prompts = len(opts.prompts)
    per_proc = (total_prompts + world_size - 1) // world_size
    start = rank * per_proc
    end = min(start + per_proc, total_prompts)
    prompts = opts.prompts[start:end]

    if rank == 0 and not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir, exist_ok=True)

    from predictor import Predictor, Config
    predictor = Predictor(Config(), num_steps=opts.num_steps, enable_training=True)

    os.makedirs(f"{opts.weights_dir}", exist_ok=True)
    weights_path = f"{opts.weights_dir}/predictor_flux_N{opts.interval}_E{opts.first_enhance}.pt"
    if os.path.exists(weights_path):
        print("Loading predictor...")
        predictor.load_model(weights_path)

    # Load model
    if opts.model_name == 'qwen-image':
        pipe = QwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image", 
            torch_dtype=torch.bfloat16,
            local_files_only=True
        ).to(device=device)
    elif opts.model_name == 'qwen-image-lightning':
        assert opts.num_steps == 8, "qwen-image-lightning only supports 8 steps."
        
        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": math.log(3),  # We use shift=3 in distillation
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": math.log(3),  # We use shift=3 in distillation
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,  # set shift_terminal to None
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)

        pipe = QwenImagePipeline.from_pretrained(
            "Qwen/Qwen-Image", 
            scheduler=scheduler,
            torch_dtype=torch.bfloat16
        ).to(device=device)

        pipe.load_lora_weights(
            "lightx2v/Qwen-Image-Lightning", weight_name="Qwen-Image-Lightning-8steps-V2.0.safetensors"
        )
    else:
        raise ValueError(f"Model name {opts.model_name} not supported.")

    from predictor import pipe_with_cache
    pipe = pipe_with_cache(pipe)

    progress_bar = tqdm(total=len(prompts) * opts.iterations, desc="Generating images") if rank == 0 else None

    idx = 0  # Image index for this process

    for _ in range(len(prompts) * opts.iterations):
        prompt = prompts[idx]
        generator = torch.Generator(device).manual_seed(int(opts.seed + idx))

        kwargs = {
            'num_steps': opts.num_steps,
            'height': opts.height,
            'width': opts.width,
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

        if opts.model_name == 'qwen-image':
            result = pipe(
                prompt=prompt,
                negative_prompt=opts.negative_prompt,
                height=opts.height,
                width=opts.width,
                num_inference_steps=opts.num_steps,
                guidance_scale=opts.guidance_scale,
                generator=generator,
                predictor=predictor,
                cache_dic=cache_dic,
                current=current,
            )
        elif opts.model_name == 'qwen-image-lightning':
            result = pipe(
                prompt=prompt,
                negative_prompt=opts.negative_prompt,
                height=1024,
                width=1024,
                num_inference_steps=opts.num_steps,
                true_cfg_scale=1.0,
                guidance_scale=opts.guidance_scale,
                generator=generator,
                predictor=predictor,
                cache_dic=cache_dic,
                current=current,
            )
        else:
            raise ValueError(f"Model name {opts.model_name} not supported.")

        images = getattr(result, 'images', None)
        if images is None:
            if isinstance(result, (list, tuple)):
                images = list(result)
            else:
                images = [result]

        for _, img in enumerate(images):
            if not isinstance(img, Image.Image):
                continue

            # Add EXIF metadata
            exif_data = Image.Exif()
            exif_data[ExifTags.Base.ImageDescription] = prompt

            filename = f"{opts.output_dir}/img_{start + idx}.jpg"
            img.save(filename, exif=exif_data, quality=95, subsampling=0)

        if rank == 0 and progress_bar is not None:
            progress_bar.update(1)

        if rank == 0:
            predictor.save_model(f"{opts.weights_dir}/predictor_flux_N{opts.interval}_E{opts.first_enhance}.pt")

        if idx == len(prompts) - 1:
            idx = 0
            continue

        idx += 1

    if rank == 0 and progress_bar is not None:
        progress_bar.close()

    dist.barrier()
    if rank == 0:
        print("All images generated.")
    dist.destroy_process_group()


def read_prompts(prompt_file: str):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the qwenimage model.")
    parser.add_argument('--prompt_file', type=str, default='prompts/train_prompts_image.txt', help='Path to the prompt text file.')
    parser.add_argument('--negative_prompt', type=str, default=" ", help='Negative prompt for guidance.')
    parser.add_argument('--width', type=int, default=1328, help='Width of the generated image.')
    parser.add_argument('--height', type=int, default=1328, help='Height of the generated image.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--model_name', type=str, default='qwen-image', choices=['qwen-image', 'qwen-image-lightning'], help='Model name.')
    parser.add_argument('--output_dir', type=str, default='samples/test', help='Directory to save images.')
    parser.add_argument('--test_FLOPs', action='store_true', help='Test inference computation cost.')
    parser.add_argument('--monitor_gpu_usage', action='store_true', help='Monitor GPU memory usage during sampling.')

    parser.add_argument('--data_prepare', type=bool, default=False)
    parser.add_argument('--data_dir', type=str, default="data")
    parser.add_argument('--weights_dir', type=str, default='predictor/ckpts')
    parser.add_argument('--phase', type=str, default='GT-Guided Training', choices=['GT-Guided Training', 'CL-AR Training'])
    parser.add_argument('--iterations', type=int, default=3)
    parser.add_argument('--interval', type=int, default=7)
    parser.add_argument('--first_enhance', type=int, default=3)

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)

    opts = SamplingOptions(
        prompts=prompts,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        model_name=args.model_name,
        output_dir=args.output_dir,
        test_FLOPs=args.test_FLOPs,
        monitor_gpu_usage=args.monitor_gpu_usage,
        data_prepare=args.data_prepare,
        data_dir=args.data_dir,
        weights_dir=args.weights_dir,
        phase=args.phase,
        iterations=args.iterations,
        interval=args.interval,
        first_enhance=args.first_enhance,
    )

    main(opts)
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train.py
