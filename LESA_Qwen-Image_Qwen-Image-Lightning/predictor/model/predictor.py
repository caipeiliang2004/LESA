import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from typing import List

from .config import Config
from .kan import KAN


class KANPredictor(nn.Module):
    
    def __init__(self, hidden_dim: int, num_steps: int, history_depth: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.history_depth = history_depth

        self.latent_cond = nn.ModuleList([
            nn.Linear(history_depth // 2, 1),
            nn.Linear(history_depth, 1),
            nn.Linear(history_depth, 1)
        ])
        self.latent_uncond = nn.ModuleList([
            nn.Linear(history_depth // 2, 1),
            nn.Linear(history_depth, 1),
            nn.Linear(history_depth, 1)
        ])

        self.timestep_cond = nn.ModuleList([
            KAN(layers_hidden=[num_steps + 1] + [hidden_dim] * 2 + [1]) for _ in range(3)
        ])
        self.timestep_uncond = nn.ModuleList([
            KAN(layers_hidden=[num_steps + 1] + [hidden_dim] * 2 + [1]) for _ in range(3)
        ])

    def forward(self, input: List[torch.Tensor], timestep: List[float], t_curr: float, module: str) -> torch.Tensor:
        
        shape, device, dtype = input[-1].shape, input[-1].device, input[-1].dtype
        if t_curr > 0.9:
            stage = 0
            K = self.history_depth // 2
        elif t_curr > 0.4:
            stage = 1
            K = self.history_depth
        else:
            stage = 2
            K = self.history_depth

        if module == 'cond':
            feat_latent = torch.stack(list(input)[-K:], dim=-1).to(device=device, dtype=dtype).reshape(-1, K)
            output_latent = self.latent_cond[stage](feat_latent).view(*shape)
            feat_timestep = torch.tensor([t_curr - t for t in list(timestep)], device=device, dtype=dtype).view(1, self.num_steps + 1)
            output_timestep = self.timestep_cond[stage](feat_timestep)
        else:
            feat_latent = torch.stack(list(input)[-K:], dim=-1).to(device=device, dtype=dtype).reshape(-1, K)
            output_latent = self.latent_uncond[stage](feat_latent).view(*shape)
            feat_timestep = torch.tensor([t_curr - t for t in list(timestep)], device=device, dtype=dtype).view(1, self.num_steps + 1)
            output_timestep = self.timestep_uncond[stage](feat_timestep)

        return input[-1] + output_timestep * output_latent


class Predictor:
    
    def __init__(self, config: Config, num_steps: int, enable_training: bool = False):
        self.config = config
        self.device = config.model.device
        self.num_steps = num_steps
        self.enable_training = enable_training
        self.model = KANPredictor(hidden_dim=config.model.kan.hidden_dim, num_steps=num_steps, history_depth=config.model.kan.history_depth).to(device=self.device, dtype=torch.bfloat16)
        
        if enable_training:
            self.model.train()
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config.solver.lr,
                weight_decay=config.solver.weight_decay
            )
            self.scheduler = StepLR(
                self.optimizer,
                step_size=config.solver.step_size,
                gamma=config.solver.gamma
            )
        else:
            self.model.eval()
    

    def load_model(self, model_path: str):
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        if self.enable_training:
            self.model.load_state_dict(checkpoint['model'])
        else:
            def strip(k: str) -> str:
                for p in ("module.", "model.", "_orig_mod."):
                    if k.startswith(p):
                        return k[len(p):]
                return k

            self.model.load_state_dict({strip(k): v for k, v in checkpoint['model'].items()} , strict=False)
        
        if self.enable_training:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])


    def save_model(self, save_path: str):
        checkpoint = {
            'config': self.config,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        
        torch.save(checkpoint, save_path)

    def train(self, pred: torch.Tensor, target: torch.Tensor):
        loss = F.l1_loss(pred, target)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()
