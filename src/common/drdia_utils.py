import bisect
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from pyteomics import mzxml, mzml
from scipy.sparse import coo_matrix

from src.common import constant
from src.utils import msg_send_utils
import re

modifications = {
    # "(UniMod:4)"
    "[Carbamidomethyl (C)]": "(UniMod:4)",
    "[Carbamidomethyl]": "(UniMod:4)",
    "[CAM]": "(UniMod:4)",
    "[+57]": "(UniMod:4)",
    "[+57.0]": "(UniMod:4)",

    # "(UniMod:26)"
    "[PCm]": "(UniMod:26)",

    # "(UniMod:5)"
    "[Carbamylation (KR)]": "(UniMod:5)",
    "[+43]": "(UniMod:5)",
    "[+43.0]": "(UniMod:5)",
    "[CRM]": "(UniMod:5)",

    # "(UniMod:7)"
    "[Deamidation (NQ)]": "(UniMod:7)",
    "[Deamidation]": "(UniMod:7)",
    "[Dea]": "(UniMod:7)",
    "[+1]": "(UniMod:7)",
    "[+1.0]": "(UniMod:7)",

    # "(UniMod:35)"
    "[Oxidation (M)]": "(UniMod:35)",
    "[Oxidation]": "(UniMod:35)",
    "[+16]": "(UniMod:35)",
    "[+16.0]": "(UniMod:35)",
    "[Oxi]": "(UniMod:35)",

    # "(UniMod:1)"
    "[Acetyl (Protein N-term)]": "(UniMod:1)",
    "[+42]": "(UniMod:1)",
    "[+42.0]": "(UniMod:1)",
    "[AAR]": "(UniMod:255)",
    "[AAS]": "(UniMod:254)",
    "[Frm]": "(UniMod:122)",
    "[+1K]": "(UniMod:1301)",
    "[+1R]": "(UniMod:1288)",
    "[PGE]": "(UniMod:27)",
    "[PGQ]": "(UniMod:28)",
    "[DTM]": "(UniMod:526)",
    "[2Ox]": "(UniMod:325)",
    "[Amn]": "(UniMod:342)",
    "[2CM]": "(UniMod:1290)",
    "[PGP]": "(UniMod:359)",
    "[NaX]": "(UniMod:30)",
    "[-2H]": "(UniMod:401)",
    "[MDe]": "(UniMod:528)",
    "[dAm]": "(UniMod:385)",
    "[Dhy]": "(UniMod:23)",
    "[Iod]": "(UniMod:129)",
    "[Lys8]": "(UniMod:259)",
    "[Arg10]": "(UniMod:267)",
    "[13C(5) 15N(1) Silac label]": "(UniMod:268)",
    "[13C(9) 15N(1) Silac label]": "(UniMod:269)",

    # "(UniMod:21)"
    "[Phosphorylation (ST)]": "(UniMod:21)",
    "[+80]": "(UniMod:21)",
    "[+80.0]": "(UniMod:21)",

    # other
    # "]M[": "][",
    # ")M(": ")(",
    "_": "",
}



def get_lib_col_dict():
    lib_col_dict = defaultdict(str)

    for key in ['transition_group_id', 'PrecursorID']:
        lib_col_dict[key] = 'transition_group_id'

    for key in ['PeptideSequence', 'Sequence', 'StrippedPeptide']:
        lib_col_dict[key] = 'PeptideSequence'

    for key in ['FullUniModPeptideName', 'ModifiedPeptide', 'LabeledSequence', 'modification_sequence',
                'ModifiedPeptideSequence']:
        lib_col_dict[key] = 'FullUniModPeptideName'

    for key in ['PrecursorCharge', 'Charge', 'prec_z']:
        lib_col_dict[key] = 'PrecursorCharge'

    for key in ['PrecursorMz', 'Q1']:
        lib_col_dict[key] = 'PrecursorMz'

    for key in ['Tr_recalibrated', 'iRT', 'RetentionTime', 'NormalizedRetentionTime', 'RT_detected']:
        lib_col_dict[key] = 'Tr_recalibrated'

    for key in ['ProductMz', 'FragmentMz', 'Q3']:
        lib_col_dict[key] = 'ProductMz'

    for key in ['FragmentType', 'FragmentIonType', 'ProductType', 'ProductIonType', 'frg_type']:
        lib_col_dict[key] = 'FragmentType'

    for key in ['FragmentCharge', 'FragmentIonCharge', 'ProductCharge', 'ProductIonCharge', 'frg_z']:
        lib_col_dict[key] = 'FragmentCharge'

    for key in ['FragmentNumber', 'frg_nr', 'FragmentSeriesNumber']:
        lib_col_dict[key] = 'FragmentNumber'

    for key in ['LibraryIntensity', 'RelativeIntensity', 'RelativeFragmentIntensity', 'RelativeFragmentIonIntensity',
                'relative_intensity']:
        lib_col_dict[key] = 'LibraryIntensity'

    # exclude
    for key in ['FragmentLossType', 'FragmentIonLossType', 'ProductLossType', 'ProductIonLossType']:
        lib_col_dict[key] = 'FragmentLossType'

    for key in ['ProteinID', 'ProteinId', 'UniprotID', 'uniprot_id', 'UniProtIds']:
        lib_col_dict[key] = 'ProteinID'

    for key in ['ProteinName', 'Protein Name', 'Protein_name', 'protein_name']:
        lib_col_dict[key] = 'ProteinName'

    for key in ['Gene', 'Genes', 'GeneName']:
        lib_col_dict[key] = 'Gene'

    for key in ['Decoy', 'decoy']:
        lib_col_dict[key] = 'decoy'

    for key in ['ExcludeFromAssay', 'ExcludeFromQuantification']:
        lib_col_dict[key] = 'ExcludeFromAssay'
    return lib_col_dict


def replace_modifications(text):
    for key, value in modifications.items():
        text = text.replace(key, value)
    return text


def relpace_isoform(text):
    result = re.sub(r'-\d+', '', text)
    result = sorted(set(result.split(';')), key=result.index)
    result = ';'.join(result)
    return result


def load_library(library_file, protein_infer_key, transition_group_id_type=1, logger=None):
    terminator = library_file.split('.')[-1]
    if terminator == 'tsv':
        library = pd.read_csv(library_file, sep="\t", engine='c')
    elif terminator in ('csv', 'txt'):
        library = pd.read_csv(library_file)
    elif terminator in ('xls', 'xlsx'):
        library = pd.read_excel(library_file)
    else:
        raise Exception("Invalid spectral library format: %s. Only .tsv and .csv formats are supported." % library_file)
    logger.info('read lib success')
    # col mapping
    lib_col_dict = get_lib_col_dict()

    for col in set(library.columns) & set(lib_col_dict.keys()):
        library.loc[:, lib_col_dict[col]] = library.loc[:, col]

    all_lib_cols = set(library.columns)
    for each_col in ['PeptideSequence', 'FullUniModPeptideName', 'PrecursorCharge', 'PrecursorMz', 'Tr_recalibrated',
                     'ProductMz', 'FragmentType', 'LibraryIntensity']:
        if each_col not in all_lib_cols:
            raise Exception("Column {} not in library.".format(each_col))

    if protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN:
        if constant.OUTPUT_COLUMN_PROTEIN not in all_lib_cols:
            raise Exception("Column ProteinID not in library.")
        if 'UniprotID' not in all_lib_cols:
            library['UniprotID'] = library[constant.OUTPUT_COLUMN_PROTEIN]
        if 'ProteinName' not in all_lib_cols:
            library['ProteinName'] = ' '
    elif protein_infer_key == constant.OUTPUT_COLUMN_PROTEIN_NAME:
        if 'ProteinName' not in all_lib_cols:
            raise Exception("Column ProteinName not in library.")
        if constant.OUTPUT_COLUMN_PROTEIN not in all_lib_cols:
            library[constant.OUTPUT_COLUMN_PROTEIN] = ' '
        if 'UniprotID' not in all_lib_cols:
            library['UniprotID'] = library[constant.OUTPUT_COLUMN_PROTEIN]

    library['UniprotID'] = library['UniprotID'].astype(str).replace(',', ';').apply(relpace_isoform)

    # if 'ExcludeFromAssay' in library.columns:
    #     library = library[library['ExcludeFromAssay'] == False]

    library['FullUniModPeptideName'] = library['FullUniModPeptideName'].astype(str)
    library['FullUniModPeptideName'] = library['FullUniModPeptideName'].apply(replace_modifications)

    if 'transition_group_id' in library.columns:
        library.drop(columns='transition_group_id', inplace=True)

    library['transition_group_id'] = library['PrecursorCharge'].astype(str) + '_' + library['FullUniModPeptideName'] + '_' + library['PrecursorCharge'].astype(str)

    # calc fragment loss
    if 'FragmentLossType' in library.columns:
        library['FragmentLossVal'] = library['FragmentLossType'].fillna('noloss').map(
            {'NH3': -17.026548, 'noloss': 0, 'H2O': -18.011113035, 'CO': -27.994915, 'N': -14.026548})

    # lib col check
    lib_cols = {'PRECURSOR_MZ_COL': 'PrecursorMz',
                'IRT_COL': 'Tr_recalibrated',
                'PRECURSOR_ID_COL': 'transition_group_id',
                'FULL_SEQUENCE_COL': 'FullUniModPeptideName',
                'PURE_SEQUENCE_COL': 'PeptideSequence',
                'PRECURSOR_CHARGE_COL': 'PrecursorCharge',
                'FRAGMENT_MZ_COL': 'ProductMz',
                'FRAGMENT_SERIES_COL': 'FragmentNumber',
                'FRAGMENT_CHARGE_COL': 'FragmentCharge',
                'FRAGMENT_TYPE_COL': 'FragmentType',
                'LIB_INTENSITY_COL': 'LibraryIntensity',
                'PROTEIN_NAME_COL': 'ProteinName',
                'DECOY_OR_NOT_COL': 'decoy'}
    necessary_columns = list(lib_cols.values())
    real_columns = list(library.columns)
    no_columns = [i for i in necessary_columns if i not in real_columns]
    if no_columns:
        logger.info("Cannot find column(s) '{}' in the spectral library.".format(";".join(no_columns)))

    logger.info('{}'.format(set(library.columns) & set(necessary_columns)))
    # assert len(set(library.columns) & set(necessary_columns)) == 13
    return lib_cols, library


def check_full_sequence(library, id_column, full_seq_column):
    abnormal_records = []
    for pep_id, full_seq in zip(library[id_column], library[full_seq_column]):
        if not pep_id.startswith("DECOY"):
            if pep_id.strip().split("_")[1] != full_seq:
                abnormal_records.append(pep_id)
    return abnormal_records


def correct_full_sequence(library, id_column, full_seq_column):
    abnormal_records = check_full_sequence(library, id_column, full_seq_column)
    abnormal_library = library[library[id_column].isin(abnormal_records)]
    abnormal_library[full_seq_column] = abnormal_library[id_column].apply(lambda x: x.strip().split("_")[1])
    new_library = library[~library[id_column].isin(abnormal_records)]
    new_library = pd.concat([new_library, abnormal_library], ignore_index=True)
    return new_library


def flatten_list(alist):
    flattened_list = []
    for elem in alist:
        flattened_list.extend(elem)
    return flattened_list


def get_precursor_indice(precursor_ids):
    precursor_indice = []
    last_record = ""
    prec_index = [0]
    for i, prec in enumerate(precursor_ids):
        if last_record != prec:
            if i:
                precursor_indice.append(prec_index)
                prec_index = [i]
        else:
            prec_index.append(i)
        last_record = prec
    precursor_indice.append(prec_index)
    return precursor_indice


def tear_library(library, lib_cols, n_threads):
    precursor_indice = get_precursor_indice(library[lib_cols["PRECURSOR_ID_COL"]])
    n_precursors = len(precursor_indice)
    n_each_chunk = n_precursors // n_threads
    chunk_indice = [[k + i * n_each_chunk for k in range(n_each_chunk)] for i in range(n_threads)]
    chunk_indice[-1].extend([i for i in range(n_each_chunk * n_threads, n_precursors)])

    return precursor_indice, chunk_indice


class MS1_Chrom:
    def __init__(self):
        self.rt_list = []
        self.spectra = []
        self.scan_list = []
        self.moz_rt_matrix = None


class MS2_Chrom:
    def __init__(self, win_id, win_min, win_max):
        self.win_id = win_id
        self.win_min = win_min
        self.win_max = win_max
        self.rt_list = []
        self.spectra = []
        self.scan_list = []
        self.moz_rt_matrix = None


def filter_spectrum(spectrum, mz_min, mz_max):
    intensity_array = spectrum['intensity array']
    mz_array = spectrum['m/z array'][intensity_array > 0]
    intensity_array = intensity_array[intensity_array > 0]
    ms_range = (mz_array >= mz_min) & (mz_array < mz_max)
    mz_array = mz_array[ms_range]
    intensity_array = intensity_array[ms_range]
    return mz_array, intensity_array


def calc_win_id(precursor_mz, win_range):
    return bisect.bisect(win_range[:, 0], precursor_mz) - 1


def load_rawdata(rawdata_file, mz_min, mz_max, rt_unit, logger=None):
    if rt_unit == 'sec':
        base_rt_multiple = 1
    elif rt_unit == 'min':
        base_rt_multiple = 60
    else:
        msg_send_utils.send_msg(msg="Invalid rt_unit: %s !\nOnly sec and min are supported!" % rt_unit)
        raise Exception("Invalid rt_unit: %s !\nOnly sec and min are supported!" % rt_unit)

    if rawdata_file.endswith(".mzXML"):
        rawdata_reader = mzxml.MzXML(rawdata_file)
        mslevel_string = "msLevel"

        def get_RT_from_rawdata_spectrum(spectrum):
            return spectrum["retentionTime"]

        def get_precursor_mz_from_rawdata_spectrum(spectrum):
            return spectrum['precursorMz'][0]['precursorMz']

        def get_winWidth_from_rawdata_spectrum(spectrum):
            return spectrum['precursorMz'][0]['windowWideness']
    elif rawdata_file.endswith(".mzML"):
        rawdata_reader = mzml.MzML(rawdata_file)
        mslevel_string = "ms level"

        def get_RT_from_rawdata_spectrum(spectrum):
            return spectrum["scanList"]["scan"][0]["scan start time"]

        def get_precursor_mz_from_rawdata_spectrum(spectrum):
            return spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]['selected ion m/z']

        def get_winWidth_from_rawdata_spectrum(spectrum):
            return spectrum["precursorList"]["precursor"][0]["isolationWindow"]["isolation window lower offset"] * 2
    else:
        raise Exception("Invalid rawdata file: %s !\nOnly mzXML and mzML files are supported!" % rawdata_file)

    def win_calculator(rawdata_reader, mslevel_string):
        raw_win = []
        flag = 0
        d1 = time.time()
        for spectrum in rawdata_reader:
            if spectrum[mslevel_string] == 1:
                flag += 1
            else:
                if flag == 0:
                    continue
                p_mz = get_precursor_mz_from_rawdata_spectrum(spectrum)
                p_width = get_winWidth_from_rawdata_spectrum(spectrum)

                raw_win.append((p_mz, p_width))

        d2 = time.time()

        raw_win = list(set(raw_win))
        raw_win.sort(key=lambda x: x[0])
        rawdata_reader.reset()
        raw_win = list(map(lambda x: [x[0] - x[1] / 2, x[0] + x[1] / 2], raw_win))
        win_range = [raw_win[0][0]]
        for i in range(len(raw_win) - 1):
            overlap = 0
            if raw_win[i][1] > raw_win[i + 1][0]:
                overlap = raw_win[i][1] - raw_win[i + 1][0]
            win_range.append(raw_win[i][1] - overlap / 2)
            win_range.append(raw_win[i + 1][0] + overlap / 2)

        win_range.append(raw_win[-1][-1])
        win_range = np.array([[win_range]]).reshape(-1, 2)
        return win_range

    t1 = time.time()
    logger.info('start calc win')
    msg_send_utils.send_msg(msg="Start calc win")
    win_range = win_calculator(rawdata_reader, mslevel_string)
    t2 = time.time()
    logger.info('end calc win. time: {}'.format(t2 - t1))
    msg_send_utils.send_msg(msg="Success calc win, time speed: {}s".format(t2 - t1))
    ms1 = MS1_Chrom()
    ms2 = [MS2_Chrom(i, each_win[0], each_win[1]) for i, each_win in enumerate(win_range)]

    rt_list = []
    last_precursor_mz, last_idx = -1, -1

    isolation_mz_list = []
    for idx, spectrum in enumerate(rawdata_reader):
        scan_id = int(spectrum['id'].split('=')[-1])
        if spectrum[mslevel_string] == 1:
            isolation_mz_list = []
            org_rt = get_RT_from_rawdata_spectrum(spectrum)
            RT = base_rt_multiple * org_rt
            mz_array, intensity_array = filter_spectrum(spectrum, mz_min, mz_max)
            ms1.rt_list.append(RT)
            ms1.spectra.append((mz_array, intensity_array))
            ms1.scan_list.append(scan_id)
            rt_list.append(RT)
            last_idx = idx

        elif spectrum[mslevel_string] == 2:
            if idx == 0:
                RT = 0
                ms1.rt_list.append(RT)
                rt_list.append(RT)
                ms1.spectra.append((np.array([0]), np.array([0])))

            precursor_mz = get_precursor_mz_from_rawdata_spectrum(spectrum)
            # 缺少ms1标记时的解析代码
            if precursor_mz in isolation_mz_list:
                RT = base_rt_multiple * get_RT_from_rawdata_spectrum(spectrum)
                ms1.rt_list.append(RT)
                rt_list.append(RT)
                ms1.spectra.append((np.array([0]), np.array([0])))
                isolation_mz_list = []
            else:
                isolation_mz_list.append(precursor_mz)

            mz_array, intensity_array = filter_spectrum(spectrum, mz_min, mz_max)
            if len(mz_array) == 0:
                last_precursor_mz = precursor_mz
                continue
            win_id = calc_win_id(precursor_mz, win_range)
            ms2[win_id].rt_list.append(RT)
            ms2[win_id].spectra.append((mz_array, intensity_array))
            ms2[win_id].scan_list.append(scan_id)
            last_precursor_mz = precursor_mz

    rt_list = list(set(rt_list))
    rt_list.sort()
    logger.info('rt_list: {}'.format(len(rt_list)))

    t3 = time.time()
    logger.info('for reader time: {}'.format(t3 - t2))
    # 清洗ms1
    if len(ms1.rt_list) < len(rt_list):
        for i in range(len(rt_list)):
            if rt_list[i] not in ms1.rt_list:
                ms1.rt_list.insert(i, rt_list[i])
                ms1.spectra.insert(i, (np.array([0]), np.array([0])))
    msg_send_utils.send_msg(msg="Clear ms1 info")
    for each_ms2 in ms2:
        if len(each_ms2.rt_list) == 1:
            continue
        else:
            rt_list_tmp, spectra_tmp = [], []
            for i in range(1, len(each_ms2.rt_list)):
                if (each_ms2.rt_list[i - 1] == each_ms2.rt_list[i]):
                    continue
                else:
                    rt_list_tmp.append(each_ms2.rt_list[i - 1])
                    spectra_tmp.append(each_ms2.spectra[i - 1])
            each_ms2.rt_list = rt_list_tmp
            each_ms2.spectra = spectra_tmp

    msg_send_utils.send_msg(msg="Clear ms2 repeat info")

    new_ms2_list = []
    new_win_range_list = []
    deal_count = 0
    for idx, each_ms2 in enumerate(ms2):
        # print('{}/{}'.format(deal_count, len(ms2)))
        deal_count = deal_count + 1
        if len(each_ms2.spectra) == 0:
            continue
        ms2_rt_list = each_ms2.rt_list
        if len(rt_list) > len(each_ms2.rt_list):
            index_array = np.where(np.isin(rt_list, ms2_rt_list))[0]
            ms2_rt_list_result = np.zeros((len(rt_list)))
            ms2_rt_list_result[index_array] = ms2_rt_list

            index_array = index_array.tolist()
            ms2_spectra_result = [(np.array([0]), np.array([0])) for _ in range(len(rt_list))]
            for pos, ms2_spectra in enumerate(each_ms2.spectra):
                ms2_spectra_result[index_array[pos]] = ms2_spectra
            each_ms2.rt_list = rt_list
            each_ms2.spectra = ms2_spectra_result
        new_ms2_list.append(each_ms2)
        new_win_range_list.append(win_range[idx])

    msg_send_utils.send_msg(msg="Clear ms2 info")
    # map rt to index
    rt_dict = defaultdict(int)
    for index in range(len(ms1.rt_list)):
        rt_dict[ms1.rt_list[index]] = index
    logger.info('start build matrix')
    msg_send_utils.send_msg(msg="Process build matrix")
    mt1 = time.time()
    for each_ms2 in new_ms2_list:
        each_ms2.moz_rt_matrix = construct_moz_rt_matrix(each_ms2, mz_max, rt_dict)
    ms1.moz_rt_matrix = construct_moz_rt_matrix(ms1, mz_max, rt_dict)
    mt2 = time.time()
    logger.info('end build matrix， time: {}'.format(mt2 - mt1))
    msg_send_utils.send_msg(msg="Finished build matrix")
    return ms1, new_ms2_list, np.array(new_win_range_list)


def construct_sparse_matrix(rt_spectra_search, mz_max, rt_dict):
    row, col, data = [], [], []
    for key, value in rt_spectra_search.items():
        col.append(rt_dict[key[0]])
        row.append(key[1])
        data.append(value)

    # 310 fix: decoy mz could exceed mz_max
    moz_rt_matrix = coo_matrix((data, (row, col)), shape=((mz_max + 310) * 1000, len(rt_dict)))
    return moz_rt_matrix


def construct_moz_rt_matrix(ms, mz_max, rt_dict):
    rt_spectra_search = construct_rt_spectra(ms.rt_list, ms.spectra)
    return construct_sparse_matrix(rt_spectra_search, mz_max, rt_dict)


def construct_rt_spectra(rt_list, spectras):
    rt_spectra_search = defaultdict(float)
    for rt, spectra in zip(rt_list, spectras):
        mz_array, intensity_array = spectra
        if len(mz_array) == 0:
            continue

        mz_array_bins = [math.floor(mz * 1000) for mz in mz_array]
        for mz_val, intensity_val in zip(mz_array_bins, intensity_array):
            rt_spectra_search[(rt, mz_val)] += intensity_val

    return rt_spectra_search
