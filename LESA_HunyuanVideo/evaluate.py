import os
import torch
import cv2
import numpy as np
import re
import argparse
from tqdm import tqdm
import lpips
from skimage.metrics import structural_similarity as ssim

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v")


def get_sorted_video_files(folder_path):
    """Get a list of video files sorted in natural order."""
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(VIDEO_EXTS)]

    def natural_sort_key(filename):
        # Perform natural sorting based on trailing digits in filename (0 if no digits)
        m = re.search(r'(\d+)(?=\.[A-Za-z0-9]+$)', filename)
        return int(m.group(1)) if m else 0

    return sorted(files, key=natural_sort_key)


def calculate_psnr(img1, img2):
    """PSNR (input is uint8 BGR)."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """SSIM (convert to grayscale)."""
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        return ssim(gray1, gray2, data_range=255)
    return ssim(img1, img2, data_range=255)


def preprocess_for_lpips(img_bgr):
    """LPIPS preprocessing: BGR->RGB, HWC[0..255] -> NCHW[-1,1]."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = img_rgb.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
    return img * 2 - 1


def evaluate_video_pair(test_video_path, ref_video_path, lpips_model, device, frame_stride=1, max_frames=None):
    """
    Evaluate a pair of videos (same filename) frame by frame and return the mean of three metrics.

    - If resolutions are different, resize reference frames to match test frames.
    - If frame counts are different, compare only up to the shorter (common) length.
    - If frame_stride > 1, subsample frames to speed up evaluation.
    """
    cap_t = cv2.VideoCapture(test_video_path)
    cap_r = cv2.VideoCapture(ref_video_path)

    if not cap_t.isOpened() or not cap_r.isOpened():
        if cap_t.isOpened():
            cap_t.release()
        if cap_r.isOpened():
            cap_r.release()
        return None

    # Compute comparable frame count (use the smaller of the two total frame counts)
    total_t = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
    total_r = int(cap_r.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(total_t, total_r)

    psnr_vals, ssim_vals, lpips_vals = [], [], []

    idx = 0
    read_count = 0
    with torch.no_grad():
        while True:
            if max_frames is not None and read_count >= max_frames:
                break

            # Frame skipping: skip stride-1 frames
            ret_t, frame_t = cap_t.read()
            ret_r, frame_r = cap_r.read()
            if not (ret_t and ret_r):
                break

            if frame_stride > 1:
                # Additionally discard stride-1 frames
                for _ in range(frame_stride - 1):
                    if not cap_t.read()[0] or not cap_r.read()[0]:
                        break
                idx += frame_stride
            else:
                idx += 1

            # Align resolution
            if frame_t.shape != frame_r.shape:
                frame_r = cv2.resize(
                    frame_r,
                    (frame_t.shape[1], frame_t.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )

            # Compute metrics
            psnr_vals.append(calculate_psnr(frame_t, frame_r))
            ssim_vals.append(calculate_ssim(frame_t, frame_r))

            img_lpips = preprocess_for_lpips(frame_t).to(device)
            ref_lpips = preprocess_for_lpips(frame_r).to(device)
            dist = lpips_model(img_lpips, ref_lpips).item()
            lpips_vals.append(dist)

            read_count += 1

            # Stop if index exceeds comparable frame count due to skipping
            if idx >= total:
                break

    cap_t.release()
    cap_r.release()

    if len(psnr_vals) == 0:
        return None

    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)),
        "frames_used": read_count,
    }


def evaluate_all_videos(test_folder, reference_folder, frame_stride=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)

    video_files = get_sorted_video_files(test_folder)

    per_video_results = []
    all_psnr, all_ssim, all_lpips = [], [], []

    for filename in tqdm(video_files, desc="Evaluating videos"):
        test_path = os.path.join(test_folder, filename)
        ref_path = os.path.join(reference_folder, filename) if reference_folder else None

        if not ref_path or not os.path.exists(ref_path):
            # Skip if reference video is missing (all these metrics are reference-based)
            continue

        result = evaluate_video_pair(
            test_video_path=test_path,
            ref_video_path=ref_path,
            lpips_model=lpips_model,
            device=device,
            frame_stride=frame_stride,
        )

        if result is not None:
            per_video_results.append((filename, result))
            all_psnr.append(result["psnr"])
            all_ssim.append(result["ssim"])
            all_lpips.append(result["lpips"])

    overall = {}
    if all_psnr:
        overall["psnr"] = float(np.mean(all_psnr))
    if all_ssim:
        overall["ssim"] = float(np.mean(all_ssim))
    if all_lpips:
        overall["lpips"] = float(np.mean(all_lpips))

    return per_video_results, overall


def main():
    parser = argparse.ArgumentParser(description='Video metrics evaluation (PSNR/SSIM/LPIPS)')
    parser.add_argument('--test_folder', type=str, required=True)
    parser.add_argument('--reference_folder', type=str, default="samples/origin")
    parser.add_argument('--frame_stride', type=int, default=1)
    args = parser.parse_args()

    per_video_results, overall = evaluate_all_videos(
        test_folder=args.test_folder,
        reference_folder=args.reference_folder,
        frame_stride=args.frame_stride,
    )

    print("\nOverall average (PSNR, SSIM, LPIPS):")
    print(f"{overall.get('psnr', 0):.3f}")
    print(f"{overall.get('ssim', 0):.4f}")
    print(f"{overall.get('lpips', 0):.4f}")


if __name__ == "__main__":
    main()
