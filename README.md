<div align=center>
  
# [CVPR 2026] *LESA*: Learnable Stage-Aware Predictors for Diffusion Model Acceleration

<p>
<a href='https://arxiv.org/abs/2602.20497'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a>
</p>

</div>

## 🔥 News
* `2026/05/27` 🚀🚀 [Code](https://github.com/caipeiliang2004/LESA) for "LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration" is released!

* `2026/02/21` 💥💥 LESA is accepted by CVPR 2026!

## 📑 Abstract
Diffusion models have achieved remarkable success in image and video generation tasks. However, the high computational demands of Diffusion Transformers (DiTs) pose a significant challenge to their practical deployment. While feature caching is a promising acceleration strategy, existing methods based on simple reusing or training-free forecasting struggle to adapt to the complex, stage-dependent dynamics of the diffusion process, often resulting in quality degradation and failing to maintain consistency with the standard denoising process. To address this, we propose a LEarnable Stage-Aware (LESA) predictor framework based on two-stage training. Our approach leverages a Kolmogorov-Arnold Network (KAN) to accurately learn temporal feature mappings from data. We further introduce a multi-stage, multi-expert architecture that assigns specialized predictors to different noise-level stages, enabling more precise and robust feature forecasting. Extensive experiments show our method achieves significant acceleration while maintaining high-fidelity generation. Experiments demonstrate 5.00x acceleration on FLUX.1-dev with minimal quality degradation (1.0% drop), 6.25x speedup on Qwen-Image with a 20.2% quality improvement over the previous SOTA (TaylorSeer), and 5.00x acceleration on HunyuanVideo with a 24.7% PSNR improvement over TaylorSeer. State-of-the-art performance on both text-to-image and text-to-video synthesis validates the effectiveness and generalization capability of our training-based framework across different models. Our code is available at [https://github.com/caipeiliang2004/LESA](https://github.com/caipeiliang2004/LESA).

## 🛠 Installation

``` cmd
git clone https://github.com/caipeiliang2004/LESA.git
```

## 📌 Citation

```bibtex
@inproceedings{lesa,
  abbr      = {CVPR},
  title     = {LESA: Learnable Stage-Aware Predictors for Diffusion Model Acceleration},
  author    = {Cai, Peiliang and Liu, Jiacheng and Xu, Haowen and Wang, Xinyu and Zou, Chang and Zhang, Lingeng},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.20497},
  note      = {to appear},
}
```

## 🧐 Contact

If you have any questions, please email [`caipeiliang2004@gmail.com`](mailto:caipeiliang2004@gmail.com).
