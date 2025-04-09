import os
from optparse import OptionParser

from src import common_config
from src.common import constant
from src.common.obj import InputParam
from src.common_logger import create_new_logger
from src.identify_process_handler import IdentifyProcessHandler
from src.utils.gpu_utils import get_usage_device
from src.utils import pkl_size_util
import multiprocessing

common_config_data = common_config.read_yml()

if __name__ == '__main__':

    multiprocessing.freeze_support()
    multiprocessing.set_start_method('spawn', force=True)

    usage_device_list, min_free_memory = get_usage_device(common_config_data['gpu']['max_member_use_rate'])
    batch_size = int(min_free_memory * 12.5)
    step_size = int(min_free_memory * 500)
    use_rate = common_config_data['gpu']['max_member_use_rate']

    parser = OptionParser(version=constant.VERSION)
    parser.add_option("--rawdata_file_dir_path", type="string", default="",
                      help="The absolute path of the mzML file to be analyzed")

    parser.add_option("--lib", type="string", default="",
                      help="The absolute path to the library file")

    parser.add_option("--out_path", type="string", default="",
                      help="The output path of the identification results; If the path does not exist, the program will automatically create it")

    parser.add_option("--decoy_method", type="string", default="mutate",
                      help="The decoy generation strategy of Identification: [mutate]")

    parser.add_option("--instrument", type="string", default="Other",
                      help="The instrument type of the mass spectrometry file: [Orbitrap exactive hf, Orbitrap exactive hf-x, Orbitrap exploris 480, Orbitrap fusion lumos, Tripletof 5600, Tripletof 6600, Other]")

    parser.add_option("--n_cycles", type="int", default="100", help="The search scope for the precursor identification")

    parser.add_option("--step_size", type="int", default=step_size,
                      help="The number of batches to be scored at each time. The higher the value, the higher the overall efficiency.")

    parser.add_option("--batch_size", type="int", default=batch_size,
                      help="The number of precursors processed per batch")

    parser.add_option("--raw_rt_unit", type="string", default="min",
                      help="Units of retention time in mass spectrometry file:[min, sec]")

    parser.add_option("--device", type="string", default="cuda",
                      help="Device: [cuda]")

    parser.add_option("--n_thread", type="int", default="5", help="Num of thread")

    parser.add_option("--fitting_rt_num", type="int", default="50000",
                      help="fitting_rt_num")

    parser.add_option("--fitting_rt_batch_size", type="int", default="100",
                      help="fitting_rt_batch_size")

    parser.add_option("--finetune_score_limit", type="float", default="0.2", help="finetune score limit")
    parser.add_option("--train_epochs", type="int", default="10", help="finetune train epochs")

    parser.add_option("--open_lib_decoy", type="int", default="1", help="open lib decoy process")
    parser.add_option("--open_identify", type="int", default="1", help="open identify process")
    parser.add_option("--open_cross_quantification", type="int", default="0", help="Cross-run quantification")

    parser.add_option("--protein_infer_key", type="string", default="ProteinID",
                      help="Protein inference key[ProteinID, ProteinName]")

    parser.add_option("--gpu_devices", type="string", default='auto',
                      help="GPU devices index list. Split by ',' ")

    (options, args) = parser.parse_args()

    if '--help' in args:
        parser.print_help()
    else:

        print(f'Usage rate less than {use_rate} device is ', usage_device_list)
        print(f'Min free memory is {min_free_memory} GB')

        input_param = InputParam()

        input_param.lib = options.lib

        input_param.n_cycles = options.n_cycles

        input_param.decoy_method = options.decoy_method

        input_param.batch_size = options.batch_size
        input_param.device = options.device
        input_param.n_thread = options.n_thread
        input_param.out_path = options.out_path
        input_param.rawdata_file_dir_path = options.rawdata_file_dir_path
        input_param.score_device = options.device
        input_param.xrm_model_file = './resource/model/base.ckpt'
        input_param.finetune_base_model_file = './resource/model/finetune_model.ckpt'
        input_param.quant_model_file = './resource/model/quant.ckpt'
        input_param.step_size = options.step_size

        input_param.protein_infer_key = options.protein_infer_key
        input_param.open_quantification = bool(options.open_cross_quantification)

        input_param.open_lib_decoy = bool(options.open_lib_decoy)
        input_param.open_identify = bool(options.open_identify)

        input_param.fitting_rt_num = 50000
        input_param.fitting_rt_batch_size = options.fitting_rt_batch_size

        input_param.raw_rt_unit = options.raw_rt_unit

        input_param.finetune_score_limit = options.finetune_score_limit
        input_param.train_epochs = options.train_epochs

        input_param.env = constant.env_linux

        input_param.instrument = options.instrument

        common_config_data = common_config.read_yml()

        input_param.n_cycles = options.n_cycles
        input_param.max_fragment = common_config_data['identify']['max_fragment']
        input_param.iso_range = common_config_data['identify']['iso_range']
        input_param.mz_min = common_config_data['identify']['mz_min']
        input_param.mz_max = common_config_data['identify']['mz_max']
        input_param.seed = common_config_data['identify']['seed']
        input_param.model_cycles = common_config_data['identify']['model_cycles']
        input_param.frag_repeat_num = common_config_data['identify']['frag_repeat_num']

        input_param.gpu_devices = options.gpu_devices
        input_param.train_pkl_size = pkl_size_util.calculate_pkl_size(min_free_memory)
        input_param.quant_pkl_size = pkl_size_util.calculate_pkl_size(min_free_memory)

        if input_param.gpu_devices == 'auto':
            if len(usage_device_list) == 0:
                print('No GPU devices available')
            else:
                try:
                    if not os.path.exists(input_param.out_path):
                        os.makedirs(input_param.out_path)
                except Exception:
                    pass
                logger, logger_file_path = create_new_logger(input_param.out_path)
                input_param.logger_file_path = logger_file_path
                idp = IdentifyProcessHandler(input_param, logger)
                idp.shell_deal_process()
        else:
            try:
                if not os.path.exists(input_param.out_path):
                    os.makedirs(input_param.out_path)
            except Exception:
                pass
            logger, logger_file_path = create_new_logger(input_param.out_path)
            input_param.logger_file_path = logger_file_path
            idp = IdentifyProcessHandler(input_param, logger)
            idp.shell_deal_process()
