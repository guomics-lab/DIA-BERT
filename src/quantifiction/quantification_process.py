import os.path
import pickle
import re
#
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from src.common import constant
from src.common import runtime_data_info
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.utils import msg_send_utils

warnings.filterwarnings('ignore')


class QuantificationProcess():

    def __init__(self, temp_lib_path, base_output, mzml_file_path, protein_infer_key='ProteinID', max_workers=5,
                 logger=None):
        self.temp_lib_path = temp_lib_path
        self.base_output = base_output
        self.mzml_files = [os.path.split(dd)[-1] for dd in mzml_file_path]
        self.protein_infer_key = protein_infer_key
        self.logger = logger

        self.max_workers = max_workers

        self.lib_iRT_flag = 'Tr_recalibrated'

    def deal_process(self):
        self.logger.info('Processing quantification')
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, status=ProgressStepStatusEnum.RUNNING,
                                msg='Processing quantification')
        try:
            if not runtime_data_info.runtime_data.current_is_success:
                msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, status=ProgressStepStatusEnum.ERROR)
                self.logger.info('Finished quant')
                return
            self.quantification_process()
        except Exception as e:
            self.logger.exception('Quantification exception')
            msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, status=ProgressStepStatusEnum.ERROR,
                                    msg='Quant exception: {}'.format(e))
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, status=ProgressStepStatusEnum.SUCCESS,
                                msg='Finished quantification')
        self.logger.info('Finished quantification')

    def relpace_isoform(self, text):
        result = re.sub(r'-\d+', '', text)
        #
        result = sorted(set(result.split(';')), key=result.index)
        result = ';'.join(result)
        return result

    def gen_dict(self, protein_id, protein_name):
        protein_id_list = protein_id.split(';')
        proteinName_list = protein_name.split(';')

        #
        #
        if self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN_NAME:
            protein_id_name_dict = dict(zip(proteinName_list, protein_id_list))
        elif self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN:
            protein_id_name_dict = dict(zip(protein_id_list, proteinName_list))

        return protein_id_name_dict

    def get_lib_info_dict(self):
        self.logger.info('Build lib info dict')
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Peak lib info')
        with open(self.temp_lib_path, mode='rb') as f:
            lib_data_s = pickle.load(f)
        lib_data = lib_data_s[1]
        if 'ProteinName' not in lib_data.columns:
            lib_data['ProteinName'] = ' '
        lib_part_irt = lib_data[
            ['transition_group_id', self.lib_iRT_flag, 'ProteinName', 'ProteinID', 'decoy', 'PrecursorCharge',
             'PeptideSequence']].drop_duplicates(
            'transition_group_id')

        lib_precursor_irt_dict = lib_part_irt.set_index('transition_group_id')[self.lib_iRT_flag].to_dict()

        #
        lib_part_irt['ProteinID'] = lib_part_irt['ProteinID'].astype(str).replace(',', ';').apply(self.relpace_isoform)
        #
        protein_id_name = lib_part_irt[lib_part_irt['decoy'] == 0][['ProteinID', 'ProteinName']].drop_duplicates(
            subset=['ProteinID', 'ProteinName'])

        for col in ['ProteinID', 'ProteinName']:
            protein_id_name[col] = protein_id_name[col].astype(str)
        protein_id_name['protein_id_name_dict'] = protein_id_name.apply(
            lambda x: self.gen_dict(x['ProteinID'], x['ProteinName']), axis=1)

        protein_id_name_dict_list = protein_id_name['protein_id_name_dict'].tolist()

        #
        protein_id_name_dict = defaultdict(str)
        for d in protein_id_name_dict_list:
            protein_id_name_dict.update({k: v for k, v in d.items() if k not in protein_id_name_dict})

        precursor_charge_dict = lib_part_irt.set_index('transition_group_id')['PrecursorCharge'].to_dict()
        precursor_seq_dict = lib_part_irt.set_index('transition_group_id')['PeptideSequence'].to_dict()
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Finished peak lib info')
        self.logger.info('Finished build lib info dict')
        return lib_precursor_irt_dict, protein_id_name_dict, precursor_charge_dict, precursor_seq_dict

    def map_col(self, df_rt):
        df_rt['transition_group_id'] = df_rt['Unnamed: 0']
        df_rt['assay_rt_kept'] = df_rt['assay_rt_kept'].astype(float)
        return df_rt

    def merge_precursor_quant_data(self, rawdata_prefix):
        precursor_file_path = os.path.join(self.base_output, rawdata_prefix, '{}_precursor.csv'.format(rawdata_prefix))
        quant_file_path = os.path.join(self.base_output, rawdata_prefix, 'quant', 'output', 'quant_sum6.csv')
        precursor_df = pd.read_csv(precursor_file_path)
        sum6_df = pd.read_csv(quant_file_path)
        sum6_df['pred_6'] = sum6_df['pred_6'].apply(lambda x: pow(x, 3))
        sum_dict = sum6_df.set_index('transition_group_id')['pred_6'].to_dict()
        precursor_df[constant.OUTPUT_COLUMN_PRECURSOR_QUANT] = precursor_df[
            constant.OUTPUT_COLUMN_PRECURSOR].apply(
            lambda x: sum_dict.get(x))
        precursor_df = precursor_df[
            [constant.OUTPUT_COLUMN_PRECURSOR, constant.OUTPUT_COLUMN_PRECURSOR_QUANT,
             constant.OUTPUT_COLUMN_PROTEIN,
             constant.OUTPUT_COLUMN_PROTEIN_NAME]]
        return precursor_df

    def build_calc_df(self):
        self.logger.info('Process read raw identify info.')
        file_col = []
        total_precursor = []
        for mzml_name in self.mzml_files:
            rawdata_prefix = mzml_name[:-5]
            precursor_file_path = os.path.join(self.base_output, rawdata_prefix,
                                               '{}_precursor.csv'.format(rawdata_prefix))
            precursor_df = pd.read_csv(precursor_file_path)
            total_precursor.extend(precursor_df[constant.OUTPUT_COLUMN_PRECURSOR].tolist())

        total_precursor = list(set(total_precursor))
        data = pd.DataFrame({'transition_group_id': total_precursor})

        file_protein_name_col_list = []
        for mzml_name in self.mzml_files:
            rawdata_prefix = mzml_name[:-5]
            precursor_quant_df = self.merge_precursor_quant_data(rawdata_prefix)
            quant_info_dict = precursor_quant_df.set_index(constant.OUTPUT_COLUMN_PRECURSOR)[
                constant.OUTPUT_COLUMN_PRECURSOR_QUANT].to_dict()

            #
            if self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN_NAME:
                info_dict = precursor_quant_df.set_index(constant.OUTPUT_COLUMN_PRECURSOR)[
                    constant.OUTPUT_COLUMN_PROTEIN_NAME].to_dict()
            elif self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN:
                info_dict = precursor_quant_df.set_index(constant.OUTPUT_COLUMN_PRECURSOR)[
                    constant.OUTPUT_COLUMN_PROTEIN].to_dict()

            data[mzml_name] = data['transition_group_id'].map(quant_info_dict)
            file_col.append(mzml_name)
            this_protein_name_col = '{}_{}'.format(rawdata_prefix, self.protein_infer_key)
            file_protein_name_col_list.append(this_protein_name_col)
            data[this_protein_name_col] = data['transition_group_id'].map(info_dict)

        data[self.protein_infer_key] = data.apply(self.get_first_non_empty_value, axis=1,
                                                  columns=file_protein_name_col_list)
        return data, file_col

    #
    def build_rt_data_df(self, data_rt):
        rt_col = []
        for mzml_name in self.mzml_files:
            rawdata_prefix = mzml_name[:-5]
            mzml_precursor_csv_path = os.path.join(self.base_output, rawdata_prefix,
                                                   '{}_precursor.csv'.format(rawdata_prefix))
            precursor_df = pd.read_csv(mzml_precursor_csv_path)
            rt_dict = precursor_df.set_index('PrecursorID')['RT'].to_dict()
            this_rt_col = '{}_rt'.format(rawdata_prefix)
            rt_col.append(this_rt_col)
            data_rt[this_rt_col] = data_rt['transition_group_id'].map(rt_dict)
        return data_rt, rt_col

    def build_irt_data_df(self, data_rt, lib_precursor_irt_dict):
        data_rt['iRT'] = data_rt['transition_group_id'].map(lib_precursor_irt_dict)
        return data_rt

    def quantification_process(self):

        lib_precursor_irt_dict, protein_id_name_dict, precursor_charge_dict, precursor_seq_dict = self.get_lib_info_dict()

        data, file_col = self.build_calc_df()

        data_clean = data[['transition_group_id', self.protein_infer_key] + file_col]
        for col in file_col:
            data_clean[col] = data_clean[col].astype(float)

        data_clean = self.tic_scale(data_clean, file_col)

        data_rt = data_clean[['transition_group_id', self.protein_infer_key] + file_col].copy()
        data_rt, rt_col = self.build_rt_data_df(data_rt)
        data_rt = self.build_irt_data_df(data_rt, lib_precursor_irt_dict)
        #
        data_rt = data_rt.sort_values(by=['iRT', 'transition_group_id'])
        data_rt['rank'] = range(len(data_rt))
        data_rt['bin'] = data_rt['rank'].apply(lambda x: x // 100)

        normRT_ret_list = []

        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Process RT normalize bin.')
        self.logger.info('Process RT normalize bin.')
        with ThreadPoolExecutor(max_workers=self.max_workers) as t:
            obj_list = []
            for bin_num in set(data_rt['bin']):
                obj = t.submit(self.norm_rt_bin, data_rt[data_rt['bin'] == bin_num], rt_col)
                obj_list.append(obj)

            for future in as_completed(obj_list):
                normRT_ret_list.append(future.result())

        data_rt = pd.concat(normRT_ret_list, axis=0)
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Finished RT normalize bin.')
        self.logger.info('Finished RT normalize bin.')


        stander_rt_col = rt_col[0]
        data_rt = data_rt.sort_values(stander_rt_col)
        data_rt['rank'] = range(len(data_rt))
        data_rt['bin'] = data_rt['rank'].apply(lambda x: x // 400)

        rt_min = min([data_rt[r].min() for r in rt_col])
        rt_max = min([data_rt[r].max() for r in rt_col])
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION,
                                msg='RT-dependent correct, RT min: {}, RT max: {}'.format(rt_min, rt_max))
        self.logger.info('RT-dependent correct, RT min: {}, RT max: {}'.format(rt_min, rt_max))
        rt_bin_value = [rt_min - 1] + data_rt[data_rt[stander_rt_col].notnull()].groupby('bin')[
            stander_rt_col].max().tolist() + [rt_max + 1]
        rt_bin_value = list(set(rt_bin_value))
        rt_bin_value.sort()
        for each_rt_col in rt_col:
            data_rt['{}_bin'.format(each_rt_col)] = pd.cut(data_rt[each_rt_col], bins=rt_bin_value).astype(str)

        ratio_ret_list = []
        rt_bin_col = [r + '_bin' for r in rt_col]
        stander_rt_bin_col = rt_bin_col[0]

        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Process RT normalize by median.')
        self.logger.info('Process RT normalize by median.')
        with ThreadPoolExecutor(max_workers=self.max_workers) as t:
            obj_list = []
            for rt_bin in set(data_rt[stander_rt_bin_col]):
                obj = t.submit(self.rt_normalize_by_median, data_rt, rt_bin, file_col, rt_bin_col)
                obj_list.append(obj)

            for future in as_completed(obj_list):
                ratio_ret_list.append(future.result())

        ratio_ret = pd.concat(ratio_ret_list, axis=0)
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Finished RT normalize by median.')
        self.logger.info('Finished RT normalize by median.')

        for index, col in enumerate(file_col):
            if index == 0:
                continue
            data_rt[col + '_ratio'] = data_rt[rt_bin_col[index]].map(ratio_ret[col])
            data_rt[col] = data_rt[col].astype(float) * data_rt[col + '_ratio'].astype(float)

        #
        df = self.gen_df(data_rt, file_col)
        df = self.clean_data(df)

        self.logger.info('Process calc maxlfq ret.')
        protein_df = self.calc_maxlfq_ret(df)
        #
        if self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN_NAME:
            #
            protein_df[constant.OUTPUT_COLUMN_PROTEIN] = protein_df[
                constant.OUTPUT_COLUMN_PROTEIN_NAME].apply(lambda x: protein_id_name_dict.get(x))
        elif self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN:
            #
            protein_df[constant.OUTPUT_COLUMN_PROTEIN_NAME] = protein_df[
                constant.OUTPUT_COLUMN_PROTEIN].apply(lambda x: protein_id_name_dict.get(x))

        save_protein_path = os.path.join(self.base_output, 'crossrun_protein.csv')
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION,
                                msg='Save crossrun protein to {}'.format(save_protein_path))
        self.logger.info('Save crossrun protein to {}'.format(save_protein_path))
        #
        save_protein_df = protein_df[protein_df[self.protein_infer_key] != 'None']
        save_protein_df.to_csv(save_protein_path, index=False)

        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Process calc crossrun precursor')
        self.logger.info('Process calc crossrun precursor')
        precursor_ret_list = []

        data_clean[self.protein_infer_key] = data_clean[self.protein_infer_key].astype(str)
        all_protein_key_list = set(data_clean[self.protein_infer_key])

        with ThreadPoolExecutor(max_workers=self.max_workers) as t:
            obj_list = []
            for protein_key in all_protein_key_list:
                obj = t.submit(self.calc_precursor, data_clean[data_clean[self.protein_infer_key] == protein_key],
                               protein_df[protein_df[self.protein_infer_key] == protein_key],
                               file_col,
                               protein_key)
                obj_list.append(obj)

            for future in as_completed(obj_list):
                precursor_ret_list.append(future.result())

        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION, msg='Finished crossrun precursor calc precursor')
        self.logger.info('Finished calc crossrun precursor')

        precursor_df = pd.concat(precursor_ret_list, axis=0)
        #
        if self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN_NAME:
            #
            precursor_df[constant.OUTPUT_COLUMN_PROTEIN] = precursor_df[
                constant.OUTPUT_COLUMN_PROTEIN_NAME].apply(lambda x: protein_id_name_dict.get(x))
        elif self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN:
            #
            precursor_df[constant.OUTPUT_COLUMN_PROTEIN_NAME] = precursor_df[
                constant.OUTPUT_COLUMN_PROTEIN].apply(lambda x: protein_id_name_dict.get(x))

        precursor_df[constant.CHARGE_COLUMN] = precursor_df['transition_group_id'].apply(
            lambda x: precursor_charge_dict.get(x))
        precursor_df[constant.PEPT_SEQ_COLUMN] = precursor_df['transition_group_id'].apply(
            lambda x: precursor_seq_dict.get(x))
        precursor_df[constant.IRT_COLUMN] = precursor_df['transition_group_id'].apply(
            lambda x: lib_precursor_irt_dict.get(x))

        precursor_df.rename(columns={'transition_group_id': constant.OUTPUT_COLUMN_PRECURSOR}, inplace=True)

        save_precursor_path = os.path.join(self.base_output, 'crossrun_precursor.csv')
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANTIFICATION,
                                msg='Save crossrun precursor to {}'.format(save_precursor_path))
        self.logger.info('Save crossrun precursor to {}'.format(save_precursor_path))
        #
        precursor_df[self.protein_infer_key] = precursor_df[self.protein_infer_key].replace('None', '')

        precursor_df.to_csv(save_precursor_path, index=False)

    def calc_precursor(self, precursor, protein, file_col, protein_key):
        try:
            #
            precursor_data = precursor[['transition_group_id'] + file_col].sort_values('transition_group_id')

            #
            mask = precursor_data[file_col].copy()
            mask = np.where(mask.notnull(), 1, np.nan)

            #
            precursor_mean = precursor_data.set_index('transition_group_id').mean(1)
            precursor_mean = pd.DataFrame(round(precursor_mean / precursor_mean.sum(), 3))
            precursor_mean.columns = ['ratio']

            # precursor
            ratio_np = np.array(precursor_mean['ratio'])
            #
            protein_quant_np = np.array(protein[file_col].values[0])

            #
            precursor = np.matmul(protein_quant_np.reshape(-1, 1), ratio_np.reshape(1, -1)).T
            precursor = np.multiply(precursor, mask)
            precursor = pd.DataFrame(data=precursor, index=precursor_mean.index.tolist(), columns=file_col)
            precursor.index.name = 'transition_group_id'
            precursor = precursor.reset_index()

            precursor[self.protein_infer_key] = protein_key
        except Exception as e:
            print(protein_key)
        return precursor

    def processParallelOld(self, df, name, margin=-10.01):
        try:
            piv_df = df[(df[self.protein_infer_key] == name) & (df['Precursor.Normalised'] > margin)]
            #
            piv = pd.pivot_table(piv_df, index='File.Name', columns='transition_group_id',
                                 values='Precursor.Normalised').sort_index()
            # fillna
            piv = piv.fillna(-1000000.0)
            temp_df = pd.DataFrame(data=self.maxlfq_solve(piv), index=piv.index.tolist(), columns=[name])
            return temp_df
        except Exception:
            self.logger.exception('calc_maxlfq_ret process exception')

    def calc_maxlfq_ret(self, df):

        self.logger.info('Process calc maxlfq ret')
        dfGrouped = df[(df['Precursor.Normalised'].notnull())].groupby(self.protein_infer_key)
        multi_res = []
        for name, group in dfGrouped:
            each_df = self.processParallelOld(group, name)
            multi_res.append(each_df.T)
        protein = pd.concat(multi_res, axis=0)
        #
        protein = protein.apply(np.exp)

        protein.index.name = self.protein_infer_key
        protein = protein.reset_index()
        self.logger.info('Finished calc maxlfq ret')
        return protein


    def maxlfq_solve(self, quantities, margin=-10.01):
        #
        samples, peptides = quantities.shape

        #
        B = np.zeros(samples)
        A = np.zeros((samples, samples))

        #
        ref = quantities.max(axis=1).values
        ref = np.where(ref > margin, ref, -np.inf)

        #
        for i in range(samples):
            for j in range(i + 1, samples):
                #
                ratios = [x - y for x, y in zip(quantities.iloc[i, :], quantities.iloc[j, :]) if
                          x > margin and y > margin]

                if len(ratios) > 0:
                    if len(ratios) >= 2:
                        ratios.sort()
                        median = ratios[len(ratios) // 2] if len(ratios) % 2 else 0.5 * (
                                ratios[(len(ratios) // 2) - 1] + ratios[len(ratios) // 2])
                    else:
                        median = ratios[0]

                    A[i, i] += 1.0
                    A[j, j] += 1.0
                    A[i, j] = A[j, i] = -1.0
                    B[i] += median
                    B[j] -= median

        #
        for i in range(samples):
            reg = 0.0001 * max(1.0, A[i, i])
            A[i, i] += reg
            B[i] += ref[i] * reg

        #
        X = np.linalg.solve(A, B)
        return X

    def gen_sr_data(self, data, file_name):
        sr_data = data[[self.protein_infer_key, 'transition_group_id', file_name]].copy()
        sr_data['File.Name'] = file_name
        sr_data.loc[:, 'Precursor.Normalised'] = sr_data[file_name]

        usecols = [self.protein_infer_key, 'transition_group_id', 'File.Name', 'Precursor.Normalised']
        return sr_data[usecols]

    def gen_df(self, data_clean, file_clos):
        #
        file_col_data_list = []
        for each_file_col in file_clos:
            each_data = self.gen_sr_data(data_clean, file_name=each_file_col)
            file_col_data_list.append(each_data)

        #
        df = pd.concat(file_col_data_list, axis=0)
        #
        for col in [self.protein_infer_key, 'transition_group_id', 'File.Name']:
            df[col] = df[col].astype(str)

        df['Precursor.Normalised'] = df['Precursor.Normalised'].astype(float)
        print(df.shape)

        df[self.protein_infer_key] = df[self.protein_infer_key].replace('nan', '').fillna('')
        return df

    def clean_data(self, df):
        # df = df[(df[self.protein_infer_key] != 'None')]

        margin = -10.0

        df['Precursor.Normalised'] = np.where(df['Precursor.Normalised'] < 1e-6, np.nan, df['Precursor.Normalised'])
        #
        df['Precursor.Normalised'] = df['Precursor.Normalised'].apply(np.log)

        #
        df['Precursor.Normalised'] = np.where(df['Precursor.Normalised'] < margin, np.nan, df['Precursor.Normalised'])

        #
        df = df[df['Precursor.Normalised'].notnull()]
        return df

    def rt_normalize_by_median(self, df, rt_bin, file_col, rt_bin_col):
        #
        try:
            df_median_value = []
            for index, col in enumerate(rt_bin_col):
                df_median_value.append(df[df[col] == rt_bin][file_col[index]].median(0))

            ratio_base = df_median_value[0]
            ratio_list = [ratio_base / q for q in df_median_value]

            ratio_dict = defaultdict(list)
            ratio_dict[rt_bin] = ratio_list
            ratio_df = pd.DataFrame.from_dict(ratio_dict).T
            ratio_df.columns = file_col
            return ratio_df
        except Exception:
            self.logger.exception('RT normalize by median exception')

    def norm_rt_bin(self, data, rt_col):
        try:
            data_rt_median_bin = pd.DataFrame(data[rt_col].median(0)).T
            stander_rt_col = rt_col[0]
            for col in rt_col[1:]:
                data_rt_median_bin[col + '_delta'] = data_rt_median_bin[stander_rt_col] - data_rt_median_bin[col]
                data.loc[:, col] = data.loc[:, col] + data_rt_median_bin[col + '_delta'].values[0]
            return data
        except Exception:
            self.logger.exception('RT normalize bin exception')
            return data

    def get_first_non_empty_value(self, row, columns):
        for col in columns:
            if pd.notna(row[col]):
                return row[col]
        return None

    def calc_log(self, x):
        return np.log(x) / np.log(2)

    def tic_scale(self, df, file_col_list):
        tic_data_list = []
        for each_file_col in file_col_list:
            tic_data = df[each_file_col].sum(0)
            tic_data_list.append(tic_data)
        tic_avg = sum(tic_data_list) / len(tic_data_list)
        for dd, each_file_col in enumerate(file_col_list):
            df[each_file_col] = df[each_file_col] * tic_avg / tic_data_list[dd]
        return df
