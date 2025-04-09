import multiprocessing
from enum import IntEnum

from src.common.model.score_model import FeatureEngineer

VERSION = 'DIA-BERT V1.0'

mz_rt_unit_list = ['ppm', 'Da']

raw_rt_unit_list = ['min', 'sec']

decoy_method_list = ['mutate']


instrument_list = [dd for dd in FeatureEngineer().instrument_s2i]

display_instrument_list = [dd.removeprefix('Orbitrap') for dd in FeatureEngineer().instrument_s2i]

env_linux = 'linux'

env_win = 'win'

env_win_shell = 'win_shell'

msg_channel = 'identify_msg'

main_msg_channel = 'main_identify_msg'

msg_queue = multiprocessing.Queue(maxsize=5000, )

QUEUE_END_FLAG = 'end'

RUNNING_COLOR = '#f8c108'

OVER_COLOR = '#27c14c'

ERROR_COLOR = '#e91b40'


class ProgressStepEnum(IntEnum):
    START = 0
    LIB_DEAL = 1
    PARSE_MZML = 2
    RT_NORMALIZATION = 3
    SCREEN = 4
    PREPARE_DATA = 5
    FINETUNE_TRAIN = 6
    FINETUNE_EVAL = 7
    QUANT = 8
    QUANTIFICATION = 9
    END = 10


class ProgressStepStatusEnum(IntEnum):
    WAIT = 1
    RUNNING = 2
    SUCCESS = 3
    ERROR = 4
    IDENTIFY_NUM = 90
    END = 99
    FAIL_END = 98
    STOPPING = 900
    STOPPED = 901
    ALL_END = 999


OUTPUT_COLUMN_PRECURSOR = 'PrecursorID'
OUTPUT_COLUMN_PEPTIDE = 'PeptideSequence'
OUTPUT_COLUMN_PROTEIN = 'ProteinID'
OUTPUT_COLUMN_PROTEIN_NAME = 'ProteinName'
OUTPUT_COLUMN_FILE_NAME = 'FileName'
OUTPUT_COLUMN_PRECURSOR_QUANT = 'PrecursorQuant'
OUTPUT_COLUMN_PROTEIN_QUANT = 'ProteinQuant'

IRT_COLUMN = 'iRT'
PEPT_SEQ_COLUMN = 'PeptideSequence'
CHARGE_COLUMN = 'PrecursorCharge'
RT_COLUMN = 'RT'

OUTPUT_PRECURSOR_COLUMN_LIST = [OUTPUT_COLUMN_PRECURSOR, OUTPUT_COLUMN_PEPTIDE,
                                OUTPUT_COLUMN_PROTEIN, OUTPUT_COLUMN_PROTEIN_NAME,
                                OUTPUT_COLUMN_FILE_NAME, OUTPUT_COLUMN_PRECURSOR_QUANT, IRT_COLUMN, RT_COLUMN]

OUTPUT_PROTEIN_COLUMN_LIST = [OUTPUT_COLUMN_PROTEIN, OUTPUT_COLUMN_PROTEIN_NAME,
                              OUTPUT_COLUMN_FILE_NAME, OUTPUT_COLUMN_PROTEIN_QUANT]


protein_infer_key_list = [OUTPUT_COLUMN_PROTEIN, OUTPUT_COLUMN_PROTEIN_NAME]

