import os
import re

import pandas as pd

from src.common import constant
from src.utils import msg_send_utils


class ResultBuildProcess():
    pass

    def __init__(self, base_output, rawdata_prefix, mzml_name, logger):
        self.base_output = base_output
        self.rawdata_prefix = rawdata_prefix
        self.mzml_name = mzml_name

        self.logger = logger

        self.model_name = 'sum6'

    def deal_process(self):
        msg_send_utils.send_msg(msg='Build result file')
        quant_path = os.path.join(self.base_output, 'quant', 'output', 'quant_{}.csv'.format(self.model_name))
        if not os.path.exists(quant_path):
            return
        quant_data = pd.read_csv(quant_path)
        quant_data.columns = ['transition_group_id', 'precursor_quant', 'file_name']
        quant_data = quant_data.drop_duplicates(subset=['transition_group_id', 'file_name'])

        precursor_fdr_path = os.path.join(self.base_output, 'finetune', 'output',
                                          'fdr_{}_eval.csv'.format(self.mzml_name))

        self.build_result_file(precursor_fdr_path, quant_data)


    def build_result_file(self, precursor_fdr_path, quant_data):
        #

        #
        precursor_fdr = pd.read_csv(precursor_fdr_path)
        precursor_fdr.columns = ['transition_group_id', 'score', 'label', 'decoy', 'q_value', 'file_name', 'iRT', 'RT']
        precursor_fdr['peptide'] = precursor_fdr['transition_group_id'].apply(self.remove_charge_modi)

        #
        file_quant_data = quant_data
        precursor_fdr['precursor_quant'] = precursor_fdr['transition_group_id'].map(
            file_quant_data.set_index('transition_group_id')['precursor_quant'])
        #
        precursor_fdr['precursor_quant'] = precursor_fdr['precursor_quant'].apply(lambda x: pow(x, 3))

        #
        protein_infer_path = os.path.join(self.base_output, '{}_protein_fdr.csv'.format(self.mzml_name))
        protein_infer = pd.read_csv(protein_infer_path)
        protein_infer['Q_Value'] = protein_infer['Q_Value'] / 2
        protein_infer = protein_infer[protein_infer['Q_Value'] <= 0.01]

        protein_name_dict = protein_infer.set_index('Protein')['ProteinName'].to_dict()

        protein_infer.to_csv(protein_infer_path, index=False)

        #
        precursor_fdr['Protein'] = precursor_fdr['peptide'].map(protein_infer.set_index('Peptides')['Protein'])
        #
        precursor_fdr['ProteinName'] = precursor_fdr['Protein'].map(protein_name_dict)
        # precursor_fdr['ProteinGroups'] = precursor_fdr['Protein'].map(protein_group_dict)
        precursor_fdr = precursor_fdr[precursor_fdr['label'] == 1]

        #
        precursor_fdr.rename(columns={'transition_group_id': constant.OUTPUT_COLUMN_PRECURSOR,
                                      'file_name': constant.OUTPUT_COLUMN_FILE_NAME,
                                      'peptide': constant.OUTPUT_COLUMN_PEPTIDE,
                                      'precursor_quant': constant.OUTPUT_COLUMN_PRECURSOR_QUANT,
                                      'Protein': constant.OUTPUT_COLUMN_PROTEIN,
                                      'ProteinName': constant.OUTPUT_COLUMN_PROTEIN_NAME}, inplace=True)

        precursor_fdr = precursor_fdr[constant.OUTPUT_PRECURSOR_COLUMN_LIST]

        precursor_fdr = precursor_fdr[precursor_fdr[constant.OUTPUT_COLUMN_PROTEIN].notnull()]

        precursor_fdr.to_csv(os.path.join(self.base_output, '{}_precursor.csv'.format(self.rawdata_prefix)),
                             index=False)

        #
        precursor_fdr = precursor_fdr.sort_values(
            by=[constant.OUTPUT_COLUMN_PROTEIN, constant.OUTPUT_COLUMN_PRECURSOR_QUANT], ascending=False)  # 降序
        protein = pd.DataFrame(precursor_fdr[precursor_fdr[constant.OUTPUT_COLUMN_PROTEIN].notnull()].groupby(
            constant.OUTPUT_COLUMN_PROTEIN).apply(
            lambda x: x[constant.OUTPUT_COLUMN_PRECURSOR_QUANT].head(3).sum()).reset_index())
        protein.columns = [constant.OUTPUT_COLUMN_PROTEIN, constant.OUTPUT_COLUMN_PRECURSOR_QUANT]
        protein[constant.OUTPUT_COLUMN_PROTEIN_NAME] = protein[constant.OUTPUT_COLUMN_PROTEIN].map(protein_name_dict)
        protein[constant.OUTPUT_COLUMN_FILE_NAME] = self.mzml_name

        #
        protein.rename(columns={constant.OUTPUT_COLUMN_PRECURSOR_QUANT: constant.OUTPUT_COLUMN_PROTEIN_QUANT}, inplace=True)
        protein = protein[constant.OUTPUT_PROTEIN_COLUMN_LIST]
        protein.to_csv(os.path.join(self.base_output, '{}_protein.csv'.format(self.rawdata_prefix)), index=False)

    def remove_charge_modi(self, text):
        text = str(text)
        result = re.sub(r'\([^)]*\)', '', text)
        result = re.sub(r'\d$', '', result)
        return result

    def remove_brackets(self, text):
        #
        result = re.sub(r'\[[^\]]*\]', '', text)
        result = re.sub(r'\-', 'X', result)
        result = re.sub(r'n', 'X', result)
        return result