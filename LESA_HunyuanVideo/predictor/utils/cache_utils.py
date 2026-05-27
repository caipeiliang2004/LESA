import os
import torch


def predict(predictor, cache_dic, current):

    with torch.set_grad_enabled(predictor.enable_training):
        pred = predictor.model(input = current['outputs'], 
                                timestep = cache_dic['timesteps'], 
                                t_curr = current['t_curr'])

        if predictor.enable_training:
            target = torch.load(f"{cache_dic['data_dir']}/gt/image_{cache_dic['idx']}/step_{current['step']}.pt", map_location='cpu').to("cuda")

            current['l1_loss'] += predictor.train(pred, target)

            if current['step'] == cache_dic['num_steps'] - 1:
                record_loss(current['l1_loss']/current['cache_steps'], cache_dic['interval'], cache_dic['first_enhance'])

    return pred.detach()


def record_loss(l1_loss: float, interval: int, first_enhance: int):

    loss_dir = os.path.join("log", f"N_{interval}_E_{first_enhance}")
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
