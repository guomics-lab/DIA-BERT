class InputParam(object):

    def __init__(self):
        self.run_env = None
        self.lib = None
        self.n_thread = None
        self.out_path = None
        self.max_fragment = None
        self.mz_min = None
        self.mz_max = None
        self.mz_unit = None
        self.mz_tol_ms1 = None
        self.mz_tol_ms2 = None
        self.seed = None
        self.iso_range = None
        self.n_cycles = None
        self.model_cycles = None
        self.rt_norm_model = None
        self.decoy_method = None
        self.frag_repeat_num = None
        self.rawdata_file_dir_path = None
        self.rt_norm_dir = None
        self.batch_size = None
        self.device = None

        self.gpu_devices = 'auto'

        self.xrm_model_file = './resource/model/base.ckpt'
        self.finetune_base_model_file = './resource/model/finetune_model.ckpt'
        self.quant_model_file = './resource/model/quant.ckpt'
        self.score_device = 'cuda'
        self.peak_group_out_path = None
        self.step_size = 25000

        self.fitting_rt_num = 50000
        self.fitting_rt_batch_size = 100

        self.queue_size = 5

        self.decoy_check = None
        self.raw_rt_unit = None
        self.skip_no_temp = 0

        self.dev_model = True

        self.only_diann_target = None
        self.use_diann_rt = None
        self.random_decoy = None

        self.identify_pkl_path = None

        self.shifting_pos = False

        self.each_parse_frag_rt_matrix_num = 100

        self.score_scan_peak_type = 0
        self.open_finetune = True
        self.open_identify = True

        self.open_base_identify = True
        self.open_lib_decoy = True

        self.open_finetune_peak = True
        self.open_finetune_train = True
        self.open_eval = True
        self.open_protein_infer = True
        self.open_quant = True

        self.draw_rt_pic = False

        self.clear_data = True

        self.finetune_score_limit = 0.2
        self.train_epochs = 10

        self.train_pkl_size = 6144
        self.quant_pkl_size = 6144

        self.open_smooth = False

        self.env = 'linux'

        self.instrument = None

        self.ext_frag_quant_open = False
        self.ext_frag_quant_model = None

        self.ext_frag_quant_fragment_num = 6
        self.ext_frag_quant_zero_type = 0
        self.ext_quant_data_open_smooth = False

        self.open_quantification = False
        self.protein_infer_key = 'ProteinID'

        self.lib_load_version = 'v6'

        self.lib_filter = True
        self.tp_version = 'v1'
        self.fitting_rt_epochs = 2

        self.logger_file_path = ''


class FinetuneParam(object):
    def __init__(self):
        pass


class IdentifyMsg(object):
    def __init__(self, mzml_name =None, mzml_index=None, step=None, status=None, msg=None):

        '''
        :param msg:
        '''
        self.mzml_name = mzml_name
        '''
        '''
        self.mzml_index = mzml_index
        '''
        @see ProgressStepEnum
        :param msg: 
        '''
        self.step = step
        '''
        @see ProgressStepStatusEnum
        '''
        self.status = status

        '''
        :param msg: 
        '''
        self.msg = msg

    @staticmethod
    def json_to_object(dct):
        return IdentifyMsg(mzml_index=dct['mzml_index'], step=dct['step'], status=dct['status'], msg=dct['msg'])

