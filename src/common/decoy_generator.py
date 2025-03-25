import multiprocessing
import os.path
import random

import numpy as np
import pandas as pd

from src.common.drdia_utils import load_library, correct_full_sequence, tear_library, flatten_list
from src.utils import msg_send_utils
from src.utils.mz_calculator import calc_fragment_mz

save_col = ['PrecursorMz', 'Tr_recalibrated', 'FullUniModPeptideName', 'PeptideSequence',
                    'PrecursorCharge', 'ProductMz', 'FragmentNumber', 'FragmentCharge', 'FragmentType',
                    'LibraryIntensity',
                    'ProteinName', 'ProteinID', 'decoy', 'transition_group_id']


def shuffle_seq(seq=None, seed=None):
    """Fisher-Yates algorithm. Modified from PECAN's decoyGenerator.py"""
    if seq is None:
        return None
    else:
        l = list(seq)
        random.seed(seed)
        for i in range(len(l) - 1, 0, -1):
            j = int(random.random() * (i + 1))
            if i == j:
                continue
            else:
                (l[i], l[j]) = (l[j], l[i])

        return tuple(l)


def reverse(seq):
    return seq[::-1]


def shift_seq(seq):
    i = len(seq) // 2
    return seq[i::] + seq[:i:]


def mutate_seq(seq):
    mutations = {"G": "L",
                 "A": "L",
                 "V": "L",
                 "L": "V",
                 "I": "V",
                 "F": "L",
                 "M": "L",
                 "P": "L",
                 "W": "L",
                 "S": "T",
                 "C": "S",
                 "T": "S",
                 "Y": "S",
                 "H": "S",
                 "K": "L",
                 "R": "L",
                 "Q": "N",
                 "E": "D",
                 "N": "Q",
                 "D": "E",
                 "U": "S"}
    seq_1 = seq[1]
    if seq_1 in mutations:
        seq_1_new = mutations[seq_1]
    else:
        seq_1_new = 'L'
    seq__2 = seq[-2]
    if seq__2 in mutations:
        seq__2_new = mutations[seq__2]
    else:
        seq__2_new = 'L'
    return [seq[0], seq_1_new] + seq[2:-2] + [seq__2_new, seq[-1]]


def get_mod_indice(sort_base):
    cursor, lock = -1, 0
    poses, mods = [], []
    for i, lett in enumerate(sort_base):
        if lett == "(":
            lock = 1
            poses.append(cursor)
            mod = ""
        elif lett == ")":
            lock = 0
            cursor -= 1
            mods.append(mod + ")")
        if not lock:
            cursor += 1
        else:
            mod += sort_base[i]
    return poses, mods


def decoy_generator_with_log(library, lib_cols, decoy_method, precursor_indice, original_colnames, result_collector,
                             fixed_colnames, seed, logger):
    try:
        decoy_generator(library, lib_cols, decoy_method, precursor_indice, original_colnames, result_collector,
                        fixed_colnames, seed)
    except Exception:
        logger.exception('decoy_generator exception')


def decoy_generator(library, lib_cols, decoy_method, precursor_indice, original_colnames, result_collector,
                    fixed_colnames, seed):
    product_mz, peptide_sequence, full_uniMod_peptide_name = [], [], []
    transition_group_id, decoy, protein_name, transition_group_id_new = [], [], [], []
    transition_name, peptide_group_label = [], []

    valid_indice = []
    library[lib_cols["PROTEIN_NAME_COL"]] = library[lib_cols["PROTEIN_NAME_COL"]].astype(str)

    for idx, pep in enumerate(precursor_indice):
        target_record = library.iloc[pep, :]

        if ("decoy" in list(library.columns)) and (list(target_record["decoy"])[0] == 1):
            continue

        valid_indice.extend(pep)

        target_fullseq = list(target_record[lib_cols["FULL_SEQUENCE_COL"]])[0]
        target_pureseq = list(target_record[lib_cols["PURE_SEQUENCE_COL"]])[0]

        if decoy_method in ["shuffle", "pseudo_reverse", "shift"]:
            unimod5, KR_end, KR_mod_end = False, False, False

            sort_base = target_fullseq[:]
            if sort_base.startswith("(UniMod:5)"):
                unimod5 = True
                sort_base = sort_base[10:]
            if sort_base[-1] in ["K", "R"]:
                KR_end = sort_base[-1]
                sort_base = sort_base[:-1]
            elif (sort_base.endswith("(UniMod:259)") or sort_base.endswith("(UniMod:267)")):
                KR_mod_end = sort_base[-13:]
                sort_base = sort_base[:-13]
            mod_indice, mod_list = get_mod_indice(sort_base)

            if KR_end or KR_mod_end:
                pure_seq_list = [i for i in target_pureseq[:-1]]
            else:
                pure_seq_list = [i for i in target_pureseq]

            seq_list = pure_seq_list[:]
            for mod_id, mod in zip(mod_indice, mod_list):
                seq_list[mod_id] += mod

            if decoy_method == "shuffle":
                shuffled_indice = shuffle_seq([i for i in range(len(seq_list))], seed=seed)
            elif decoy_method == "pseudo_reverse":
                shuffled_indice = reverse([i for i in range(len(seq_list))])
            elif decoy_method == "shift":
                shuffled_indice = shift_seq([i for i in range(len(seq_list))])
            decoy_fullseq = "".join([seq_list[i] for i in shuffled_indice])
            decoy_pureseq = "".join([pure_seq_list[i] for i in shuffled_indice])

            if unimod5:
                decoy_fullseq = "(UniMod:5)" + decoy_fullseq
            if KR_end:
                decoy_fullseq += KR_end
                decoy_pureseq += KR_end
            elif KR_mod_end:
                decoy_fullseq += KR_mod_end
                decoy_pureseq += KR_mod_end[0]

        elif decoy_method == "reverse":
            unimod5 = False

            sort_base = target_fullseq[:]
            if sort_base.startswith("(UniMod:5)"):
                unimod5 = True
                sort_base = sort_base[10:]

            mod_indice, mod_list = get_mod_indice(sort_base)

            pure_seq_list = [i for i in target_pureseq]

            seq_list = pure_seq_list[:]
            for mod_id, mod in zip(mod_indice, mod_list):
                seq_list[mod_id] += mod

            shuffled_indice = reverse([i for i in range(len(seq_list))])
            decoy_fullseq = "".join([seq_list[i] for i in shuffled_indice])
            decoy_pureseq = "".join([pure_seq_list[i] for i in shuffled_indice])

            if unimod5:
                decoy_fullseq = "(UniMod:5)" + decoy_fullseq

        elif decoy_method == "mutate":
            unimod5 = False

            sort_base = target_fullseq[:]
            if sort_base.startswith("(UniMod:5)"):
                unimod5 = True
                sort_base = sort_base[10:]

            mod_indice, mod_list = get_mod_indice(sort_base)

            pure_seq_list = [i for i in target_pureseq]
            mutated_pure_seq_list = mutate_seq(pure_seq_list)

            mutated_seq_list = mutated_pure_seq_list[:]
            for mod_id, mod in zip(mod_indice, mod_list):
                mutated_seq_list[mod_id] += mod

            decoy_fullseq = "".join(mutated_seq_list)
            decoy_pureseq = "".join(mutated_pure_seq_list)

            if unimod5:
                decoy_fullseq = "(UniMod:5)" + decoy_fullseq

        for charge, tp, series in zip(target_record[lib_cols["FRAGMENT_CHARGE_COL"]],
                                      target_record[lib_cols["FRAGMENT_TYPE_COL"]],
                                      target_record[lib_cols["FRAGMENT_SERIES_COL"]]):
            product_mz.append(calc_fragment_mz(decoy_fullseq, decoy_pureseq, charge, "%s%d" % (tp, series)))
            peptide_sequence.append(decoy_pureseq)
            full_uniMod_peptide_name.append(decoy_fullseq)

        if "transition_name" in original_colnames:
            transition_name.extend(["DECOY_" + list(target_record["transition_name"])[0]] * target_record.shape[0])
        if "PeptideGroupLabel" in original_colnames:
            peptide_group_label.extend(
                ["DECOY_" + list(target_record["PeptideGroupLabel"])[0]] * target_record.shape[0])

        transition_group_id.extend(
            ["DECOY_" + list(target_record[lib_cols["PRECURSOR_ID_COL"]])[0]] * target_record.shape[0])

        transition_group_id_new.extend(
            ["DECOY_" + list(target_record[lib_cols["PRECURSOR_ID_COL"]])[0]] * target_record.shape[0])
        decoy.extend([1] * target_record.shape[0])
        protein_name.extend(["DECOY_" + list(target_record[lib_cols["PROTEIN_NAME_COL"]])[0]] * target_record.shape[0])

    result_collector.append([product_mz, peptide_sequence, full_uniMod_peptide_name,
                             transition_group_id, decoy, protein_name, transition_name,
                             peptide_group_label, library.iloc[valid_indice, :].loc[:, fixed_colnames]])


def filter_library(library, lib_cols, mz_min, mz_max, lib_filter, lib_load_version):
    if lib_filter:

        if lib_load_version == 'v5':
            library = library[
                (library[lib_cols["PRECURSOR_MZ_COL"]] >= mz_min) & (library[lib_cols["PRECURSOR_MZ_COL"]] < mz_max)]
            library = library[
                (library[lib_cols["FRAGMENT_MZ_COL"]] >= mz_min) & (library[lib_cols["FRAGMENT_MZ_COL"]] < mz_max)]

        # 序列长度在>=7 <= 30
        library = library[(library['PeptideSequence'].str.len() >= 7) & (library['PeptideSequence'].str.len() <= 30)]
        # charge >=2 <=4
        library = library[(library['PrecursorCharge'] >= 2) & (library['PrecursorCharge'] <= 4)]
        # fragment num >= 4
        library = library.groupby('transition_group_id').filter(lambda x: len(x) >= 4)

        # 最多保留20个fragment
        library = library.sort_values(by=['transition_group_id', 'LibraryIntensity'], ascending=[True, False])
        library = library.groupby('transition_group_id').head(20).reset_index(drop=True)

        if lib_load_version == 'v6':
            library = library[
                (library[lib_cols["PRECURSOR_MZ_COL"]] >= mz_min) & (library[lib_cols["PRECURSOR_MZ_COL"]] < mz_max)]
            library = library[
                (library[lib_cols["FRAGMENT_MZ_COL"]] >= mz_min) & (library[lib_cols["FRAGMENT_MZ_COL"]] < mz_max)]
    # return library[save_col]
    return library



def load_lib(lib, n_threads, seed, mz_min, mz_max, decoy_method,
                           logger, protein_infer_key, lib_filter):
    lib_cols, library = load_library(lib, protein_infer_key, logger=logger)
    library = filter_library(library, lib_cols, mz_min, mz_max, lib_filter)
    return lib_cols, library

def generate_decoys_thread(lib, n_threads, seed, mz_min, mz_max, decoy_method,
                           logger, protein_infer_key, lib_filter, lib_load_version):
    output_filename = os.path.join(os.path.dirname(lib),
                                   os.path.basename(lib) + ".with_decoys_{}_{}.tsv".format(lib_filter, lib_load_version))


    lib_cols, library = load_library(lib, protein_infer_key, logger=logger)
    library = filter_library(library, lib_cols, mz_min, mz_max, lib_filter, lib_load_version)
    library = correct_full_sequence(library, lib_cols["PRECURSOR_ID_COL"], lib_cols["FULL_SEQUENCE_COL"])

    library = library[
        (library[lib_cols["PRECURSOR_MZ_COL"]] >= mz_min) & (library[lib_cols["PRECURSOR_MZ_COL"]] < mz_max)]
    library = library[
        (library[lib_cols["FRAGMENT_MZ_COL"]] >= mz_min) & (library[lib_cols["FRAGMENT_MZ_COL"]] < mz_max)]
    library.index = [i for i in range(library.shape[0])]
    logger.info('after moz library shape: {}'.format(len(library)))
    msg_send_utils.send_msg(msg='After moz library shape: {}'.format(len(library)))

    library.index = [i for i in range(library.shape[0])]
    logger.info('final library shape: {}'.format(len(library)))
    msg_send_utils.send_msg(msg='Final library shape: {}'.format(len(library)))
    precursor_indice, chunk_indice = tear_library(library, lib_cols, n_threads)

    original_colnames = list(library.columns)
    modifiable_colnames = [lib_cols["FRAGMENT_MZ_COL"],
                           lib_cols["PURE_SEQUENCE_COL"],
                           lib_cols["FULL_SEQUENCE_COL"],
                           lib_cols["PRECURSOR_ID_COL"],
                           lib_cols["PROTEIN_NAME_COL"],
                           "transition_name", "decoy", "PeptideGroupLabel"]
    fixed_colnames = [i for i in original_colnames if i not in modifiable_colnames]
    logger.info('original_colnames : {}'.format(original_colnames))

    generators = []
    mgr = multiprocessing.Manager()
    result_collectors = [mgr.list() for _ in range(n_threads)]

    logger.info('chunk_indice start')
    msg_send_utils.send_msg(msg='Chunk indice start')
    for i, chunk_index in enumerate(chunk_indice):
        precursor_index = [precursor_indice[idx] for idx in chunk_index]
        p = multiprocessing.Process(target=decoy_generator_with_log,
                                    args=(library, lib_cols, decoy_method, precursor_index, original_colnames,
                                          result_collectors[i], fixed_colnames, seed, logger))
        generators.append(p)
        p.daemon = True
        p.start()

    for p in generators:
        p.join()
    msg_send_utils.send_msg(msg='Finished Chunk indice')

    # for i, chunk_index in enumerate(chunk_indice):
    #     precursor_index = [precursor_indice[idx] for idx in chunk_index]
    #     decoy_generator(library, lib_cols, decoy_method, precursor_index, original_colnames, result_collectors, fixed_colnames, seed)

    product_mz = flatten_list([collector[0][0] for collector in result_collectors])
    peptide_sequence = flatten_list([collector[0][1] for collector in result_collectors])
    full_uniMod_peptide_name = flatten_list([collector[0][2] for collector in result_collectors])
    transition_group_id = flatten_list([collector[0][3] for collector in result_collectors])
    decoy = flatten_list([collector[0][4] for collector in result_collectors])
    protein_name = flatten_list([collector[0][5] for collector in result_collectors])
    transition_name = flatten_list([collector[0][6] for collector in result_collectors])
    peptide_group_label = flatten_list([collector[0][7] for collector in result_collectors])
    fixed_part = pd.concat([collector[0][8] for collector in result_collectors])

    modified_part = pd.DataFrame({lib_cols["FRAGMENT_MZ_COL"]: product_mz,
                                  lib_cols["PURE_SEQUENCE_COL"]: peptide_sequence,
                                  lib_cols["FULL_SEQUENCE_COL"]: full_uniMod_peptide_name,
                                  lib_cols["PRECURSOR_ID_COL"]: transition_group_id,
                                  lib_cols["DECOY_OR_NOT_COL"]: decoy,
                                  lib_cols["PROTEIN_NAME_COL"]: protein_name})
    if "transition_name" in original_colnames:
        modified_part["transition_name"] = transition_name
    if "PeptideGroupLabel" in original_colnames:
        modified_part["PeptideGroupLabel"] = peptide_group_label

    modified_part.index = [nn for nn in range(modified_part.shape[0])]
    fixed_part.index = [nn for nn in range(fixed_part.shape[0])]

    if "decoy" in original_colnames:
        decoy_data = pd.concat([modified_part, fixed_part], axis=1).loc[:, original_colnames]
    else:
        decoy_data = pd.concat([modified_part, fixed_part], axis=1).loc[:, original_colnames + ["decoy"]]
        library["decoy"] = [0 for _ in range(library.shape[0])]

    same_seq_list = list(
        set(library[lib_cols["PURE_SEQUENCE_COL"]].unique()) & set(decoy_data[lib_cols["PURE_SEQUENCE_COL"]].unique()))
    if len(same_seq_list) > 0:
        decoy_data = decoy_data[~decoy_data[lib_cols["PURE_SEQUENCE_COL"]].isin(same_seq_list)]

    library_with_decoys = pd.concat([library, decoy_data])
    library_with_decoys = library_with_decoys.sort_values(
        by=[lib_cols["PRECURSOR_ID_COL"], lib_cols["LIB_INTENSITY_COL"]], ascending=[True, False])
    library_with_decoys.index = [i for i in range(library_with_decoys.shape[0])]

    # fix ProductMz of decoy
    if 'FragmentLossVal' in library.columns:
        library_with_decoys['ProductMz'] = np.where(library_with_decoys["decoy"] == 0, \
                                                    library_with_decoys['ProductMz'], \
                                                    library_with_decoys['ProductMz'] + library_with_decoys[
                                                        'FragmentLossVal'])

    msg_send_utils.send_msg(msg='Save decoy lib to {}'.format(output_filename))
    library_with_decoys.to_csv(output_filename, index=False, sep="\t")
    return lib_cols, library_with_decoys
