

def calculate_pkl_size(actual_memory_gb):
    #
    base_memory_gb = 40
    base_pkl_size = 6144

    #
    scale_factor = base_pkl_size / base_memory_gb

    #  pkl size
    actual_pkl_size = actual_memory_gb * scale_factor

    #
    actual_pkl_size = int(actual_pkl_size)
    actual_pkl_size = (actual_pkl_size + 7) // 8 * 8

    return actual_pkl_size
