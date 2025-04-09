'''

'''


def list_split(data_arr, each_num):
    return [data_arr[i: i + each_num] for i in range(0, len(data_arr), each_num)]


def divide_list(data_arr, n):
    k, m = divmod(len(data_arr), n)
    return (data_arr[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

