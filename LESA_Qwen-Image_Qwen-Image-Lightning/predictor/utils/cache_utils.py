import os
import torch


def predict(predictor, cache_dic, current):

    with torch.set_grad_enabled(predictor.enable_training):
        pred = predictor.model(input = current['outputs'][current['module']], 
                                timestep = cache_dic['timesteps'], 
                                t_curr = current['t_curr'],
                                module = current['module'])

        if predictor.enable_training:
            target = torch.load(f"{cache_dic['data_dir']}/gt/image_{cache_dic['idx']}/{current['module']}/step_{current['step']}.pt", map_location='cpu').to("cuda")

            current['l1_loss'][current['module']] += predictor.train(pred, target)

            if current['step'] == cache_dic['num_steps'] - 1:
                record_loss(current['l1_loss'][current['module']]/current['cache_steps'], cache_dic['interval'], cache_dic['first_enhance'], current['module'])

    return pred.detach()


def record_loss(l1_loss: float, interval: int, first_enhance: int, module: str):

    loss_dir = os.path.join("log", f"N_{interval}_E_{first_enhance}/{module}")
    os.makedirs(loss_dir, exist_ok=True)
    iteration = 1

    while True:
        filename = f"iteration_{iteration}.txt"
        filepath = os.path.join(loss_dir, filename)
        
        if not os.path.exists(filepath):
            break
            
        with open(filepath, 'r') as f:
            line_count = sum(1 for _ in f)
        
        if line_count < 100:
            break
            
        iteration += 1
        
    with open(filepath, 'a') as f:
        f.write(f"{l1_loss:<36.18f}\n")


def pipe_with_cache(pipe):

    import types
    from models.transformers.transformer_qwenimage import QwenImageTransformer2DModel as LocalQwenImageTransformer2DModel

    pipe.transformer.forward = types.MethodType(LocalQwenImageTransformer2DModel.forward, pipe.transformer)

    return pipe
