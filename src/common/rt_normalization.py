import os
import os.path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ropwr import RobustPWRegression

matplotlib.use('pdf')
from sklearn.preprocessing import KBinsDiscretizer



def read_rt_model_params(out_file_dir):
    try:
        with open(os.path.join(out_file_dir, 'time_points.txt'), 'r') as f:
            data_list = f.readlines()
        return len(data_list)
    except Exception:
        return 0


def get_minmax_rt(out_file_dir, file_name='time_points.txt'):
    try:
        rt_list = []
        with open(os.path.join(out_file_dir, file_name), 'r') as f:
            for line in f:
                irt, rt = line.strip().split("\t")
                rt_list.append(float(rt))
        return min(rt_list), max(rt_list)
    except Exception:
        return None, None


'''
'''


def get_tutorials_param(irt_list, rt_list):
    strategy = 'uniform'
    n_bins = 10
    pw = RobustPWRegression(objective="huber", degree=1, continuous_deriv=False,
                            monotonic_trend="ascending", reg_l1=0, reg_l2=0, h_epsilon=1)

    irt_list = np.array(irt_list)
    rt_list = np.array(rt_list)

    est = KBinsDiscretizer(n_bins=n_bins, strategy=strategy)
    est.fit(irt_list.reshape(-1, 1), rt_list)
    splits = est.bin_edges_[0][1:-1]
    pw.fit(irt_list, rt_list, splits=splits)
    return pw, splits


def fit_irt_model_by_data_v5(irt_rt_list, out_file_dir, draw_pic=False, file_name='time_points.txt'):
    irt_list, rt_list, = [], []
    if irt_rt_list is None:
        with open(os.path.join(out_file_dir, 'time_points.txt'), 'r') as f:
            irt_rt_lines = f.readlines()
            for line_data in irt_rt_lines:
                irt, rt = line_data.strip().split('\t')
                irt_list.append(float(irt))
                rt_list.append(float(rt))
    else:
        if draw_pic:
            with open(os.path.join(out_file_dir, 'time_points.txt'), 'w+') as f:
                for irt, rt in irt_rt_list:
                    irt_list.append(float(irt))
                    rt_list.append(float(rt))
                    f.write('%.5f\t%.2f\n' % (float(irt), float(rt)))
        else:
            for irt, rt in irt_rt_list:
                irt_list.append(float(irt))
                rt_list.append(float(rt))

    pw, splits = get_tutorials_param(irt_list, rt_list)
    try:
        if draw_pic:
            line_X = np.arange(min(irt_list) - 20, max(irt_list) + 20)
            line_y = pw.predict(line_X)
            plt.figure(figsize=(6, 6))
            plt.scatter(irt_list, rt_list)
            for s in splits:
                plt.axvline(s, color="grey", linestyle="--")

            plt.plot(line_X, line_y, c="red")
            plt.xlabel("iRT")
            plt.ylabel("RT by DIA-BERT")
            plt.title("DIA-BERT RT normalization, {}".format(os.path.split(out_file_dir)[-1]))
            plt.savefig(os.path.join(out_file_dir, "{}_irt_model_tutorials.pdf".format(file_name)))
    except Exception:
        pass
    return pw
