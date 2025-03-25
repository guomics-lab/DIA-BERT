import GPUtil
import torch


#
def get_top_free_device(max_num):
    if torch.cuda.is_available():
        gpu_free_info = get_gpu_usage()
        gpu_free_info.sort(key=lambda x: x['memoryFree'], reverse=True)
        gpu_free_info = gpu_free_info[:max_num]
        topn_min_free = gpu_free_info[-1]['memoryFree'] / 1024
        topn_device_list = [str(nn['deviceID']) for nn in gpu_free_info]
    else:
        return [], 0

    return topn_device_list, topn_min_free


#
def get_usage_device(useRate=0.5):
    usage_device_list = []
    min_free = 0
    if torch.cuda.is_available():
        gpu_free_info = get_gpu_usage()
        gpu_free_info.sort(key=lambda x: x['memoryFree'], reverse=True)
        for nn in gpu_free_info:
            if nn['useRate'] < useRate:
                usage_device_list.append(nn['deviceID'])
                min_free = nn['memoryFree'] / 1024
    else:
        return [], 0

    return usage_device_list, min_free

def get_gpu_usage():
    gpus = GPUtil.getGPUs()
    gpu_usages = []
    for gpu in gpus:
        gpu_usages.append({'memoryFree': gpu.memoryFree, 'deviceID': gpu.id, 'useRate': gpu.memoryUtil })
    return gpu_usages

get_gpu_usage()
