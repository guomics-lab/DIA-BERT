from src.common.model.score_dataset import Dataset
from src.common.model.score_model import DIABertModel

# do not delete
none_d = Dataset(None)


def load_model(model_file_path, device='cpu'):
    model = DIABertModel().load_f16_model(model_file_path)
    return model.to(device)
