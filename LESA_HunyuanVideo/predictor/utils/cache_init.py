from collections import deque

def cache_init(**kwargs):
    
    cache = {}
    cache[-1] = {}

    cache_dic = {}
    cache_dic['cache'] = cache
    cache_dic['num_steps'] = kwargs['num_steps']
    cache_dic['timesteps'] = None
    cache_dic['height'] = kwargs['height']
    cache_dic['width'] = kwargs['width']
    cache_dic['test_FLOPs'] = kwargs['test_FLOPs']
    cache_dic['monitor_gpu_usage'] = kwargs['monitor_gpu_usage']
    cache_dic['data_prepare'] = kwargs['data_prepare']
    cache_dic['data_dir'] = kwargs['data_dir']
    cache_dic['phase'] = kwargs['phase']
    cache_dic['idx'] = kwargs['idx']

    cache_dic['interval'] = kwargs['interval']
    cache_dic['first_enhance'] = kwargs['first_enhance']

    current = {}
    current['step'] = 0
    current['cache_counter'] = 0
    current['cache_steps'] = 0
    current['is_first_steps'] = False
    current['l1_loss'] = 0.0
    current['outputs'] = deque([None] * 8, maxlen=8)
    current['t_curr'] = -1
    

    return cache_dic, current