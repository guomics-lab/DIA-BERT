import os
import pickle
import time

from src.common import decoy_generator
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.utils import msg_send_utils


class LibProcess():

    def __init__(self, lib_path, decoy_method, mz_min, mz_max, seed, thread_num, lib_filter, logger):

        self.lib_path = lib_path
        self.decoy_method = decoy_method
        self.mz_min = mz_min
        self.mz_max = mz_max
        self.seed = seed
        self.thread_num = thread_num

        self.lib_prefix = os.path.split(self.lib_path)[-1].split('.')[0]

        self.lib_filter = lib_filter
        self.logger = logger

    def deal_process(self, protein_infer_key, lib_load_version):
        msg_send_utils.send_msg(step=ProgressStepEnum.LIB_DEAL, status=ProgressStepStatusEnum.RUNNING, msg='Processing load lib, {}'.format(self.lib_path))
        lib_cols, library, temp_lib_path = self.only_load_lib(protein_infer_key, lib_load_version)
        pr_id_count = library[lib_cols["PRECURSOR_ID_COL"]].nunique()
        msg_send_utils.send_msg(step=ProgressStepEnum.LIB_DEAL, status=ProgressStepStatusEnum.SUCCESS, msg='Finished load lib, precursor count: {}'.format(pr_id_count))
        return lib_cols, library, temp_lib_path


    '''
    '''
    def only_load_lib(self, protein_infer_key, lib_load_version='v6'):
        times = time.time()

        self.lib_prefix = os.path.split(self.lib_path)[-1].split('.')[0]
        lib_path = os.path.split(self.lib_path)[0]

        temp_lib_name = '{}_{}_{}_{}_{}_{}.pkl'.format(self.lib_prefix, self.decoy_method, self.mz_min, self.mz_max,
                                                       self.lib_filter, lib_load_version)

        temp_lib_path = os.path.join(lib_path, temp_lib_name)
        if not os.path.exists(temp_lib_path):
            self.logger.info('lib temp {} not exist, read lib. {}'.format(temp_lib_path, self.lib_path))
            msg_send_utils.send_msg(msg='Lib temp not exist, read lib. {}'.format(self.lib_path))
            if self.decoy_method != 'no':
                lib_cols, library = decoy_generator.generate_decoys_thread(self.lib_path, self.thread_num, self.seed, self.mz_min, \
                                                                    self.mz_max, self.decoy_method, self.logger, protein_infer_key, self.lib_filter, lib_load_version)
            else:
                lib_cols, library = decoy_generator.load_lib(self.lib_path, self.thread_num, self.seed, self.mz_min, \
                                                                    self.mz_max, self.decoy_method, self.logger, protein_infer_key, self.lib_filter)
                if 'decoy' not in library.columns:
                    #
                    self.logger.info('Lib columns do not contain decoy, add decoy 0')
                    msg_send_utils.send_msg(msg='Lib columns do not contain decoy, add decoy')
                    library['decoy'] = 0
            library = library[
                (library[lib_cols['PRECURSOR_MZ_COL']] <= self.mz_max) & (library[lib_cols['PRECURSOR_MZ_COL']] >= self.mz_min)]
            library = library[
                (library[lib_cols['FRAGMENT_MZ_COL']] <= self.mz_max) & (library[lib_cols['FRAGMENT_MZ_COL']] >= self.mz_min)]
            library[lib_cols["PRECURSOR_ID_COL"]] = library.apply(lambda x: self.calc_pr(x[lib_cols["PRECURSOR_ID_COL"]], x['decoy']), axis=1)

            #
            with open(temp_lib_path, 'wb') as f:
                msg_send_utils.send_msg(msg='Save lib temp pkl to {}'.format(temp_lib_path))
                pickle.dump((lib_cols, library), f)
        else:
            self.logger.info('lib temp exist, load lib. {}'.format(temp_lib_path))
            msg_send_utils.send_msg(msg='Lib temp exist, load lib, read lib pkl. {}'.format(temp_lib_path))
            with open(temp_lib_path, 'rb') as f:
                lib_cols, library = pickle.load(f)

        t1 = time.time()
        self.logger.info('read: {}'.format(t1 - times))
        return lib_cols, library, temp_lib_path

    '''
    clac pr
    '''
    def calc_pr(self, transition_group_id, decoy):
        if decoy == 0:
            return ''.join(transition_group_id.split('_')[-2:])
        else:
            return 'DECOY_' + ''.join(transition_group_id.split('_')[-2:])
