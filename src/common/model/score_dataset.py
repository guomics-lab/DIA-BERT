
import torch.utils.data as pt_data

class Dataset(pt_data.Dataset):
    def __init__(self, file_name):
        self.file_name = file_name
        self.rsm = None
        self.file = None
        self.label = None
        self.precursor_id = None
        self.feat = None
        self.frag_info = None

    def __getitem__(self, idx):
        return_dict = {"file": self.file[idx],
                       "rsm": self.rsm[idx],
                       "label": self.label[idx],
                       "precursor_id": self.precursor_id[idx],
                       "feat": self.feat[idx],
                       "frag_info": self.frag_info[idx]}
        return return_dict

    def __len__(self):
        return len(self.precursor_id)

    def fit_scale(self, scaler_list, embedding=False):
        pass