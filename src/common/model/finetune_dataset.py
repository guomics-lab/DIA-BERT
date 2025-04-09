
import torch.utils.data as pt_data

class Dataset(pt_data.Dataset):
    def __init__(self):
        self.rsm = None
        self.frag_info = None
        self.feat = None
        self.label = None
        self.file = None
        self.precursor_id = None

    def __getitem__(self, idx):
        return_dict = {"rsm": self.rsm[idx],
                       "frag_info": self.frag_info[idx],
                       "feat": self.feat[idx],
                       "label": self.label[idx],
                       "file": self.file[idx],
                       "precursor_id": self.precursor_id[idx]}
        return return_dict

    def __len__(self):
        return len(self.precursor_id)

    def fit_scale(self):
        pass