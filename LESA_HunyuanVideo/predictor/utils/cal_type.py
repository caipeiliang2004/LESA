def cal_type(cache_dic, current):

    current['is_first_steps'] = (current['step'] < cache_dic['first_enhance'])

    if current['is_first_steps'] or (current['cache_counter'] == cache_dic['interval'] - 1):
        current['type'] = 'full'
        current['cache_counter'] = 0
    
    else:
        current['type'] = 'cache'
        current['cache_counter'] += 1
        current['cache_steps'] += 1

    if current['step'] == 3:
        current['type'] = 'full'
