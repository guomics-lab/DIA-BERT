import os
import pickle
import re

import numpy as np
import pandas as pd
from collections import defaultdict

from src.common import runtime_data_info, constant
from src.common.constant import ProgressStepEnum, ProgressStepStatusEnum
from src.pyproteininference.pipeline import ProteinInferencePipeline
from src.utils import msg_send_utils


class ProteinInferProcess():

    def __init__(self, temp_lib_path, protein_infer_key, base_raw_out_dir=None, mzml_name=None, logger=None):
        self.config = './config/proteinInfer.yaml'
        self.mzml_name = mzml_name
        self.protein_infer_key = protein_infer_key
        self.base_out_path = base_raw_out_dir
        self.temp_lib_path = temp_lib_path

        self.logger = logger

    def deal_process(self):
        #
        msg_send_utils.send_msg(step=ProgressStepEnum.QUANT, status=ProgressStepStatusEnum.RUNNING,
                                msg='Processing quant')
        try:
            if not runtime_data_info.runtime_data.current_is_success:
                msg_send_utils.send_msg(step=ProgressStepEnum.QUANT, status=ProgressStepStatusEnum.ERROR)
                return
            precursor_fdr_path = os.path.join(self.base_out_path, 'finetune', 'output',
                                              'fdr_{}_eval.csv'.format(self.mzml_name))
            if not os.path.exists(precursor_fdr_path):
                self.logger.error('Precursor fdr file is not exist, {}'.format(precursor_fdr_path))
                #
                msg_send_utils.send_msg(step=ProgressStepEnum.QUANT, status=ProgressStepStatusEnum.ERROR,
                                        msg='Precursor fdr file is not exist, {}'.format(precursor_fdr_path))
                runtime_data_info.runtime_data.current_is_success = False
                return
            df = pd.read_csv(precursor_fdr_path)
            if df.empty:
                return
            precursor_protein_dict, protein_id_name_dict = self.read_peptide_proteins()
            self.protein_inference(precursor_fdr_path, precursor_protein_dict, protein_id_name_dict)
        except Exception as e:
            self.logger.exception('Protein infer exception')
            runtime_data_info.runtime_data.current_is_success = False
            msg_send_utils.send_msg(step=ProgressStepEnum.QUANT, status=ProgressStepStatusEnum.ERROR,
                                    msg='Protein infer exception: {}'.format(e))

    def protein_inference(self, precursor_fdr_path, precursor_protein_dict, protein_id_name_dict):
        #
        df_with_protein_ids = self.update_file_with_proteins(precursor_fdr_path, precursor_protein_dict)
        #
        proteinID_file_path = os.path.join(self.base_out_path, f'{self.mzml_name}_proteinID.txt')
        if os.path.exists(proteinID_file_path):
            os.remove(proteinID_file_path)

        with open(proteinID_file_path, "a+") as f:
            f.write('\t'.join(df_with_protein_ids.columns) + "\n")
            for index, row in df_with_protein_ids.iterrows():
                line = '\t'.join([str(row['PSMId']), str(row['score']), str(row['peptide']),
                                  str(row['proteinIds']).replace(';', '\t')])
                f.write(line + "\n")

        #
        out_file_path = os.path.join(self.base_out_path, f'{self.mzml_name}_protein_fdr.csv')
        pipeline = ProteinInferencePipeline(parameter_file=self.config,
                                            combined_files=proteinID_file_path,
                                            output_filename=out_file_path)
        pipeline.execute()
        #
        protein_fdr_df = pd.read_csv(out_file_path)

        protein_fdr_df = protein_fdr_df[(protein_fdr_df['Protein'] != '') & (protein_fdr_df['Protein'].notnull())]
        protein_fdr_df['Protein'] = protein_fdr_df['Protein'].astype(str)

        protein_fdr_df = protein_fdr_df[(~protein_fdr_df['Protein'].str.startswith('DECOY'))]

        if self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN:
            protein_fdr_df[constant.OUTPUT_COLUMN_PROTEIN_NAME] = protein_fdr_df['Protein'].apply(lambda x: protein_id_name_dict.get(x))

        elif self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN_NAME:
            protein_fdr_df[constant.OUTPUT_COLUMN_PROTEIN] = protein_fdr_df['Protein'].apply(lambda x: protein_id_name_dict.get(x))
            protein_fdr_df.rename(columns={'Protein': constant.OUTPUT_COLUMN_PROTEIN_NAME}, inplace=True)
            protein_fdr_df.rename(columns={constant.OUTPUT_COLUMN_PROTEIN: 'Protein'}, inplace=True)

        protein_fdr_df.to_csv(out_file_path, index=False)

    def get_protein_name(self, protein_name):
        if protein_name:
            return str(protein_name).removeprefix('DECOY_')
        return protein_name

    def relpace_isoform(self, text):
        result = re.sub(r'-\d+', '', text)
        #
        result = sorted(set(result.split(';')), key=result.index)
        result = ';'.join(result)
        return result

    def gen_dict(self, protein_id, protein_name):
        protein_id_list = protein_id.split(';')
        proteinName_list = protein_name.split(';')

        if self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN_NAME:
            protein_id_name_dict = dict(zip(proteinName_list, protein_id_list))
        elif self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN:
            protein_id_name_dict = dict(zip(protein_id_list, proteinName_list))

        return protein_id_name_dict

    def read_peptide_proteins(self):

        with open(self.temp_lib_path, mode='rb') as f:
            lib_data_s = pickle.load(f)
        lib_data = lib_data_s[1]

        lib_part_irt = lib_data[['transition_group_id', 'ProteinName', 'ProteinID', 'decoy']].drop_duplicates('transition_group_id')
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

        if self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN:
            precursor_protein_dict = lib_part_irt.set_index('transition_group_id')['ProteinID'].to_dict()
        elif self.protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN_NAME:
            precursor_protein_dict = lib_part_irt.set_index('transition_group_id')['ProteinName'].to_dict()

        return precursor_protein_dict, protein_id_name_dict

    def update_file_with_proteins(self, file_path, precursor_protein_dict):
        df = pd.read_csv(file_path, usecols=['transition_group_id', 'score', 'decoy', 'iRT', 'RT'])
        df['score'] = df['score'].astype(float)
        df['decoy'] = df['decoy'].astype(int)

        df['peptide'] = df['transition_group_id'].apply(self.remove_charge_modi).apply(
            lambda x: x.removeprefix('DECOY'))

        df['proteinIds'] = df['transition_group_id'].map(precursor_protein_dict).astype(str)
        df['proteinIds'] = np.where(df['decoy'] == 1, df['proteinIds'].apply(self.add_decoy_proteinIds),
                                    df['proteinIds'])

        ret = df.groupby(by=['peptide', 'proteinIds'])['score'].sum().reset_index()
        ret.columns = ['peptide', 'proteinIds', 'score']

        ret['PSMId'] = ret.reset_index().index + 1
        return ret[['PSMId', 'score', 'peptide', 'proteinIds']]

    def remove_charge_modi(self, text):
        result = re.sub(r"\([^()]*\)|\[.*?\]|[^A-Z]", '', str(text))
        return result

    def add_decoy_proteinIds(self, proteinIds):
        proteinIds_list = proteinIds.split(';')
        proteinIds_list = ['DECOY_' + p for p in proteinIds_list]
        return ';'.join(proteinIds_list)
