import numpy as np

from src.common.model.score_model import FeatureEngineer


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_mzml_nearest_rt(mzml_rt):
    fe = FeatureEngineer()
    rt_list = list(fe.rt_s2i.keys())
    mzml_rt = find_nearest(rt_list, mzml_rt)
    return mzml_rt

