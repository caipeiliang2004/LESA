import os
from dataclasses import dataclass
import torch
import torch.distributed as dist
from einops import rearrange
from PIL import ExifTags, Image
from tqdm import tqdm

from flux.sampling import get_noise, get_schedule, prepare, unpack, denoise_cache
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5


@dataclass
class SamplingOptions:
    prompts: list[str]          # List of prompts
    width: int                  # Image width
    height: int                 # Image height
    num_steps: int              # Number of sampling steps
    guidance: float             # Guidance value
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
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if model_name == "flux-schnell":
        assert opts.num_steps == 4, "flux-schnell only supports 4 steps"

    # Load T5 and CLIP models to GPU
    if rank == 0:
        print("Loading models...")
    t5 = load_t5(device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(device)

    # Load model to GPU
    model = load_flow_model(model_name, device=device)
    ae = load_ae(model_name, device=device)

    progress_bar = tqdm(total=len(prompts) * opts.iterations, desc="Generating images") if rank == 0 else None

    idx = 0  # Image index for this process

    for _ in range(len(prompts) * opts.iterations):
        prompt = prompts[idx]
        seed = int(opts.seed + idx)

        # Prepare input
        if opts.data_prepare:
            assert opts.interval == 1 and opts.iterations == 1, "Data preparation only supports interval 1 and iterations 1"
            x = get_noise(1, opts.height, opts.width, device=device, dtype=torch.bfloat16, seed=seed)
            os.makedirs(f"{opts.data_dir}/noise", exist_ok=True)
            torch.save(x, f"{opts.data_dir}/noise/noise_{idx}.pt")
        else:
            assert os.path.exists(f"{opts.data_dir}/noise/noise_{idx}.pt"), f"Noise file {opts.data_dir}/noise/noise_{idx}.pt does not exist"
            x = torch.load(f"{opts.data_dir}/noise/noise_{idx}.pt", map_location='cpu').to(device)

        inp = prepare(t5, clip, x, prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(model_name != "flux-schnell")) # type: ignore
        
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

        # Denoising
        with torch.no_grad():
            x = denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, cache_dic=cache_dic, current=current, predictor=predictor)
                
            # Decode latent variables
            x = unpack(x.float(), opts.height, opts.width)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                x = ae.decode(x)

        # Convert to PIL format and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x, "b c h w -> b h w c")

        img = Image.fromarray((127.5 * (x[0] + 1.0)).cpu().byte().numpy())

        exif_data = Image.Exif()
        exif_data[ExifTags.Base.ImageDescription] = prompt
        
        fn = os.path.join(opts.output_dir, f"img_{idx}.jpg")
        img.save(fn, exif=exif_data, quality=95, subsampling=0)

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

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument('--prompt_file', type=str, default='prompts/train_prompts_image.txt', help='Path to the prompt text file.')
    parser.add_argument('--width', type=int, default=1024, help='Width of the generated image.')
    parser.add_argument('--height', type=int, default=1024, help='Height of the generated image.')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of sampling steps.')
    parser.add_argument('--guidance', type=float, default=3.5, help='Guidance value.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--model_name', type=str, default='flux-dev', choices=['flux-schnell', 'flux-dev'], help='Model name.')
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
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
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
    # PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 src/train.py
