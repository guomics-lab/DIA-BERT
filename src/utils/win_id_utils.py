
'''

'''

def split_win_id_list(pmt_win_id_list, step=10):
    win_id_pos_arr = []
    start_pos = 0
    end_pos = 0
    start_val = pmt_win_id_list[start_pos]
    for i in range(len(pmt_win_id_list)):
        end_pos = i
        if pmt_win_id_list[i] > start_val + step:
            win_id_pos_arr.append([start_pos, end_pos])
            start_pos = end_pos
            start_val = pmt_win_id_list[start_pos]
    win_id_pos_arr.append([start_pos, end_pos + 1])
    return win_id_pos_arr




