import copy
import collections
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy
import statistics


from src.pyproteininference import parameters
from src.pyproteininference import in_silico_digest
from src.pyproteininference import reader as pi_reader
from src.pyproteininference import datastore as pi_datastore
from src.pyproteininference import scoring as pi_scoring
from src.pyproteininference import export as pi_export
from src.pyproteininference import inference as pi_inference
from src.pyproteininference.inference import Inference
from src.pyproteininference.pipeline import ProteinInferencePipeline

logger = logging.getLogger(__name__)

# set up our logger
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class HeuristicPipeline(ProteinInferencePipeline):
    """
    This is the Protein Inference Heuristic class which houses the logic to run the Protein Inference Heuristic method
     to determine the best inference method for the given data.
    Logic is executed in the [execute][pyproteininference.heuristic.HeuristicPipeline.execute] method.

    Attributes:
        parameter_file (str): Path to Protein Inference Yaml Parameter File.
        database_file (str): Path to Fasta database used in proteomics search.
        target_files (str/list): Path to Target Psm File (Or a list of files).
        decoy_files (str/list): Path to Decoy Psm File (Or a list of files).
        combined_files (str/list): Path to Combined Psm File (Or a list of files).
        target_directory (str): Path to Directory containing Target Psm Files.
        decoy_directory (str): Path to Directory containing Decoy Psm Files.
        combined_directory (str): Path to Directory containing Combined Psm Files.
        output_directory (str): Path to Directory where output will be written.
        output_filename (str): Path to Filename where output will be written. Will override output_directory.
        id_splitting (bool): True/False on whether to split protein IDs in the digest.
            Advanced usage only.
        append_alt_from_db (bool): True/False on whether to append
            alternative proteins from the DB digestion in Reader class.
        pdf_filename (str): Filepath to be written to by Heuristic Plotting method.
            This is optional and a default filename will be created in output_directory if this is left as None.
        inference_method_list (list): List of inference methods used in heuristic determination.
        datastore_dict (dict): Dictionary of [DataStore][pyproteininference.datastore.DataStore]
            objects generated in heuristic determination with the inference method as the key of each entry.
        selected_methods (list): a list of String representations of the selected inference methods based on the
            heuristic.
        selected_datastores (dict):
            a Dictionary of [DataStore object][pyproteininference.datastore.DataStore] objects as selected by the
            heuristic.
        output_type (str): How to output results. Can either be "all" or "optimal". Will either output all results
            or will only output the optimal results.

    """

    RATIO_CONSTANT = 2
    OUTPUT_TYPES = ["all", "optimal"]

    def __init__(
        self,
        parameter_file=None,
        database_file=None,
        target_files=None,
        decoy_files=None,
        combined_files=None,
        target_directory=None,
        decoy_directory=None,
        combined_directory=None,
        output_directory=None,
        output_filename=None,
        id_splitting=False,
        append_alt_from_db=True,
        pdf_filename=None,
        output_type="all",
    ):
        """

        Args:
            parameter_file (str): Path to Protein Inference Yaml Parameter File.
            database_file (str): Path to Fasta database used in proteomics search.
            target_files (str/list): Path to Target Psm File (Or a list of files).
            decoy_files (str/list): Path to Decoy Psm File (Or a list of files).
            combined_files (str/list): Path to Combined Psm File (Or a list of files).
            target_directory (str): Path to Directory containing Target Psm Files.
            decoy_directory (str): Path to Directory containing Decoy Psm Files.
            combined_directory (str): Path to Directory containing Combined Psm Files.
            output_directory (str): Path to Directory where output will be written.
            output_filename (str): Path to Filename where output will be written.
                Will override output_directory.
            id_splitting (bool): True/False on whether to split protein IDs in the digest.
                Advanced usage only.
            append_alt_from_db (bool): True/False on whether to append alternative proteins
                from the DB digestion in Reader class.
            pdf_filename (str): Filepath to be written to by Heuristic Plotting method.
                This is optional and a default filename will be created in output_directory if this is left as None
            output_type (str): How to output results. Can either be "all" or "optimal". Will either output all results
                        or will only output the optimal results.

        Returns:
            HeuristicPipeline: [HeuristicPipeline][pyproteininference.heuristic.HeuristicPipeline] object

        Example:
            >>> heuristic = pyproteininference.heuristic.HeuristicPipeline(
            >>>     parameter_file=yaml_params,
            >>>     database_file=database,
            >>>     target_files=target,
            >>>     decoy_files=decoy,
            >>>     combined_files=combined_files,
            >>>     target_directory=target_directory,
            >>>     decoy_directory=decoy_directory,
            >>>     combined_directory=combined_directory,
            >>>     output_directory=dir_name,
            >>>     output_filename=output_filename,
            >>>     append_alt_from_db=append_alt,
            >>>     pdf_filename=pdf_filename,
            >>>     output_type="all"
            >>> )
        """

        self.parameter_file = parameter_file
        self.database_file = database_file
        self.target_files = target_files
        self.decoy_files = decoy_files
        self.combined_files = combined_files
        self.target_directory = target_directory
        self.decoy_directory = decoy_directory
        self.combined_directory = combined_directory
        self.output_directory = output_directory
        self.output_filename = output_filename
        self.id_splitting = id_splitting
        self.append_alt_from_db = append_alt_from_db
        self.output_type = output_type
        if self.output_type not in self.OUTPUT_TYPES:
            raise ValueError("The variable output_type must be set to either 'all' or 'optimal'")
        if not pdf_filename:
            if self.output_directory and not self.output_filename:
                self.pdf_filename = os.path.join(self.output_directory, "heuristic_plot.pdf")
            elif self.output_filename:
                self.pdf_filename = os.path.join(os.path.split(self.output_filename)[0], "heuristic_plot.pdf")
            else:
                self.pdf_filename = os.path.join(os.getcwd(), "heuristic_plot.pdf")

        else:
            self.pdf_filename = pdf_filename

        self.inference_method_list = [
            Inference.INCLUSION,
            Inference.EXCLUSION,
            Inference.PARSIMONY,
            Inference.PEPTIDE_CENTRIC,
        ]
        self.datastore_dict = {}
        self.selected_methods = None
        self.selected_datastores = {}

        self._validate_input()

        self._set_output_directory()

        self._log_append_alt_from_db()

    def execute(self, fdr_threshold=0.05):
        """
        This method is the main driver of the heuristic method.
        This method calls other classes and methods that make up the heuristic pipeline.
        This includes but is not limited to:

        1. Loops over the main inference methods: Inclusion, Exclusion, Parsimony, and Peptide Centric.
        2. Determines the optimal inference method based on the input data as well as the database file.
        3. Outputs the results and indicates the optimal results.

        Args:
            fdr_threshold (float): The Qvalue/FDR threshold the heuristic method uses to base calculations from.

        Returns:
            None:

        Example:
            >>> heuristic = pyproteininference.heuristic.HeuristicPipeline(
            >>>     parameter_file=yaml_params,
            >>>     database_file=database,
            >>>     target_files=target,
            >>>     decoy_files=decoy,
            >>>     combined_files=combined_files,
            >>>     target_directory=target_directory,
            >>>     decoy_directory=decoy_directory,
            >>>     combined_directory=combined_directory,
            >>>     output_directory=dir_name,
            >>>     output_filename=output_filename,
            >>>     append_alt_from_db=append_alt,
            >>>     pdf_filename=pdf_filename,
            >>>     output_type="all"
            >>> )
            >>> heuristic.execute(fdr_threshold=0.05)

        """

        pyproteininference_parameters = parameters.ProteinInferenceParameter(
            yaml_param_filepath=self.parameter_file
        )

        digest = in_silico_digest.PyteomicsDigest(
            database_path=self.database_file,
            digest_type=pyproteininference_parameters.digest_type,
            missed_cleavages=pyproteininference_parameters.missed_cleavages,
            reviewed_identifier_symbol=pyproteininference_parameters.reviewed_identifier_symbol,
            max_peptide_length=pyproteininference_parameters.restrict_peptide_length,
            id_splitting=self.id_splitting,
        )
        if self.database_file:
            logger.info("Running In Silico Database Digest on file {}".format(self.database_file))
            digest.digest_fasta_database()
        else:
            logger.warning(
                "No Database File provided, Skipping database digest and only taking protein-peptide mapping from the "
                "input files."
            )

        for inference_method in self.inference_method_list:

            method_specific_parameters = copy.deepcopy(pyproteininference_parameters)

            logger.info("Overriding inference type {}".format(method_specific_parameters.inference_type))

            method_specific_parameters.inference_type = inference_method

            logger.info("New inference type {}".format(method_specific_parameters.inference_type))
            logger.info("FDR Threshold Set to {}".format(method_specific_parameters.fdr))

            reader = pi_reader.GenericReader(
                target_file=self.target_files,
                decoy_file=self.decoy_files,
                combined_files=self.combined_files,
                parameter_file_object=method_specific_parameters,
                digest=digest,
                append_alt_from_db=self.append_alt_from_db,
            )
            reader.read_psms()

            data = pi_datastore.DataStore(reader=reader, digest=digest)

            data.restrict_psm_data()

            data.recover_mapping()

            data.create_scoring_input()

            if method_specific_parameters.inference_type == Inference.EXCLUSION:
                data.exclude_non_distinguishing_peptides()

            score = pi_scoring.Score(data=data)
            score.score_psms(score_method=method_specific_parameters.protein_score)

            if method_specific_parameters.picker:
                data.protein_picker()
            else:
                pass

            pi_inference.Inference.run_inference(data=data, digest=digest)

            data.calculate_q_values()

            self.datastore_dict[inference_method] = data

        self.selected_methods = self.determine_optimal_inference_method(
            false_discovery_rate_threshold=fdr_threshold, pdf_filename=self.pdf_filename
        )
        self.selected_datastores = {x: self.datastore_dict[x] for x in self.selected_methods}

        if self.output_type == "all":
            self._write_all_results(parameters=method_specific_parameters)
        elif self.output_type == "optimal":
            self._write_optimal_results(parameters=method_specific_parameters)
        else:
            self._write_optimal_results(parameters=method_specific_parameters)

    def generate_roc_plot(self, fdr_max=0.2, pdf_filename=None):
        """
        This method produces a PDF ROC plot overlaying the 4 inference methods apart of the heuristic algorithm.

        Args:
            fdr_max (float): Max FDR to display on the plot.
            pdf_filename (str): Filename to write roc plot to.

        Returns:
            None:

        """
        f = plt.figure()
        for inference_method in self.datastore_dict.keys():
            fdr_vs_target_hits = self.datastore_dict[inference_method].generate_fdr_vs_target_hits(fdr_max=fdr_max)
            fdrs = [x[0] for x in fdr_vs_target_hits]
            target_hits = [x[1] for x in fdr_vs_target_hits]
            plt.plot(fdrs, target_hits, '-', label=inference_method.replace("_", " "))
            target_fdr = self.datastore_dict[inference_method].parameter_file_object.fdr
            if inference_method in self.selected_methods:
                best_value = min(fdrs, key=lambda x: abs(x - target_fdr))
                best_index = fdrs.index(best_value)
                best_target_hit_value = target_hits[best_index]  # noqa F841

        plt.axvline(target_fdr, color="black", linestyle='--', alpha=0.75, label="Target FDR")
        plt.legend()
        plt.xlabel('Decoy FDR')
        plt.ylabel('Target Protein Hits')
        plt.xlim([-0.01, fdr_max])
        plt.legend(loc='lower right')
        plt.title("FDR vs Target Protein Hits per Inference Method")
        if pdf_filename:
            logger.info("Writing ROC plot to: {}".format(pdf_filename))
            f.savefig(pdf_filename)
        plt.close()

    def _write_all_results(self, parameters):
        """
        Internal method that loops over all results and writes them out.
        """
        for method in list(self.datastore_dict.keys()):
            datastore = self.datastore_dict[method]
            if method in self.selected_methods:
                inference_method_string = "{}_{}".format(method, "optimal_method")
            else:
                inference_method_string = method
            if not self.output_filename and self.output_directory:
                # If a filename is not provided then construct one using output_directory
                # Note: output_directory will always get set even if its set as None - gets set to cwd
                inference_filename = os.path.join(
                    self.output_directory,
                    "{}_{}_{}_{}_{}".format(
                        inference_method_string,
                        parameters.tag,
                        datastore.short_protein_score,
                        datastore.psm_score,
                        "protein_inference_results.csv",
                    ),
                )
            if self.output_filename:
                # If the user specified an output filename then split it apart and insert the inference method
                # Then reconstruct the file
                split = os.path.split(self.output_filename)
                path = split[0]
                filename = split[1]
                inference_filename = os.path.join(path, "{}_{}".format(inference_method_string, filename))
            export = pi_export.Export(data=self.datastore_dict[method])
            export.export_to_csv(
                output_filename=inference_filename,
                directory=self.output_directory,
                export_type=parameters.export,
            )

    def _write_optimal_results(self, parameters):
        """
        Internal method that writes out the optimized results.
        """

        for method in self.selected_methods:
            datastore = self.datastore_dict[method]
            inference_method_string = "{}_{}".format(method, "optimal_method")
            if not self.output_filename and self.output_directory:
                # If a filename is not provided then construct one using output_directory
                # Note: output_directory will always get set even if its set as None - gets set to cwd
                inference_filename = os.path.join(
                    self.output_directory,
                    "{}_{}_{}_{}_{}".format(
                        inference_method_string,
                        parameters.tag,
                        datastore.short_protein_score,
                        datastore.psm_score,
                        "protein_inference_results.csv",
                    ),
                )
            if self.output_filename:
                # If the user specified an output filename then split it apart and insert the inference method
                # Then reconstruct the file
                split = os.path.split(self.output_filename)
                path = split[0]
                filename = split[1]
                inference_filename = os.path.join(path, "{}_{}".format(inference_method_string, filename))
            export = pi_export.Export(data=self.selected_datastores[method])
            export.export_to_csv(
                output_filename=inference_filename,
                directory=self.output_directory,
                export_type=parameters.export,
            )

    def determine_optimal_inference_method(
        self,
        false_discovery_rate_threshold=0.05,
        upper_empirical_threshold=1,
        lower_empirical_threshold=0.5,
        pdf_filename=None,
    ):
        """
        This method determines the optimal inference method from Inclusion, Exclusion, Parsimony, Peptide-Centric.

        Args:
            false_discovery_rate_threshold (float): The fdr threshold to use in heuristic algorithm -
                This parameter determines the maximum fdr used when creating a range of finite FDR values.
            upper_empirical_threshold (float): Upper Threshold used for parsimony/peptide centric cutoff for
                the heuristic algorithm.
            lower_empirical_threshold (float): Lower Threshold used for inclusion/exclusion cutoff for
                the heuristic algorithm.
            pdf_filename (str): Filename to write heuristic density plot to.


        Returns:
            list: List of string representations of the recommended inference methods.

        """

        # Get the number of passing proteins
        number_stdev_from_mean_dict = {}
        fdrs = [false_discovery_rate_threshold * 0.01 * x for x in range(100)]
        for fdr in fdrs:
            stdev_from_mean = self.determine_number_stdev_from_mean(false_discovery_rate=fdr)
            number_stdev_from_mean_dict[fdr] = stdev_from_mean

        stdev_collection = collections.defaultdict(list)
        for fdr in fdrs:
            for key in number_stdev_from_mean_dict[fdr]:
                stdev_collection[key].append(number_stdev_from_mean_dict[fdr][key])

        heuristic_scores = self.generate_density_plot(
            number_stdevs_from_mean=stdev_collection, pdf_filename=pdf_filename
        )

        # Apply conditional statement with lower and upper thresholds
        if (
            heuristic_scores[Inference.PARSIMONY] <= lower_empirical_threshold
            or heuristic_scores[Inference.PEPTIDE_CENTRIC] <= lower_empirical_threshold
        ):
            # If parsimony or peptide centric are less than the lower empirical threshold
            # Then select the best method of the two
            logger.info(
                "Either parsimony {} or peptide centric {} pass empirical threshold {}. "
                "Selecting the best method of the two.".format(
                    heuristic_scores[Inference.PARSIMONY],
                    heuristic_scores[Inference.PEPTIDE_CENTRIC],
                    lower_empirical_threshold,
                )
            )
            sub_dict = {
                Inference.PARSIMONY: heuristic_scores[Inference.PARSIMONY],
                Inference.PEPTIDE_CENTRIC: heuristic_scores[Inference.PEPTIDE_CENTRIC],
            }

            if (
                heuristic_scores[Inference.PARSIMONY] <= lower_empirical_threshold
                and heuristic_scores[Inference.PEPTIDE_CENTRIC] <= lower_empirical_threshold
            ):
                # If both are under the threshold return both
                selected_methods = [Inference.PARSIMONY, Inference.PEPTIDE_CENTRIC]

            else:
                selected_methods = [min(sub_dict, key=sub_dict.get)]

        # If the above condition does not apply
        elif (
            heuristic_scores[Inference.EXCLUSION] <= upper_empirical_threshold
            or heuristic_scores[Inference.INCLUSION] <= upper_empirical_threshold
        ):
            # If exclusion or inclusion are less than the upper empirical threshold
            # Then select the best method of the two
            logger.info(
                "Either inclusion {} or exclusion {} pass empirical threshold {}. "
                "Selecting the best method of the two.".format(
                    heuristic_scores[Inference.INCLUSION],
                    heuristic_scores[Inference.EXCLUSION],
                    upper_empirical_threshold,
                )
            )
            sub_dict = {
                Inference.EXCLUSION: heuristic_scores[Inference.EXCLUSION],
                Inference.INCLUSION: heuristic_scores[Inference.INCLUSION],
            }

            if (
                heuristic_scores[Inference.EXCLUSION] <= upper_empirical_threshold
                and heuristic_scores[Inference.INCLUSION] <= upper_empirical_threshold
            ):
                # If both are under the threshold return both
                selected_methods = [Inference.INCLUSION, Inference.EXCLUSION]

            else:
                selected_methods = [min(sub_dict, key=sub_dict.get)]

        else:
            # If we have no conditional scenarios...
            # Select the best method
            logger.info("No methods pass empirical thresholds, selecting the best method")
            selected_methods = [min(heuristic_scores, key=heuristic_scores.get)]

        logger.info("Method(s) {} selected with the heuristic algorithm".format(", ".join(selected_methods)))
        return selected_methods

    def generate_density_plot(self, number_stdevs_from_mean, pdf_filename=None):
        """
        This method produces a PDF Density Plot plot overlaying the 4 inference methods part of the heuristic algorithm.

        Args:
            number_stdevs_from_mean (dict): a dictionary of the number of standard deviations from the mean per
                inference method for a range of FDRs.
            pdf_filename (str): Filename to write heuristic density plot to.

        Returns:
            dict: a dictionary of heuristic scores per inference method which correlates to the
                maximum point of the density plot per inference method.

        """
        f = plt.figure()

        heuristic_scores = {}
        for method in number_stdevs_from_mean:
            readible_method_name = Inference.INFERENCE_NAME_MAP[method]
            kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40, ec="k", label=readible_method_name)
            x, y, _ = plt.hist(number_stdevs_from_mean[method], **kwargs)
            center = y[list(x).index(max(x))]
            heuristic_scores[method] = abs(center)

        plt.axvline(0, color="black", linestyle='--', alpha=0.75)
        plt.title("Density Plot of the Number of Standard Deviations from the Mean")
        plt.xlabel('Number of Standard Deviations from the Mean')
        plt.ylabel('Number of Observations')
        plt.legend(loc='upper right')
        if pdf_filename:
            logger.info("Writing Heuristic Density plot to: {}".format(pdf_filename))
            f.savefig(pdf_filename)
        else:
            plt.show()
        plt.close()

        logger.info("Heuristic Scores")
        logger.info(heuristic_scores)

        return heuristic_scores

    def determine_number_stdev_from_mean(self, false_discovery_rate):
        """
        This method calculates the mean of the number of proteins identified at a specific FDR of all
        4 methods and then for each method calculates the number of standard deviations
        from the previous calculated mean.

        Args:
            false_discovery_rate (float): The false discovery rate used as a cutoff for calculations.

        Returns:
            dict: a dictionary of the number of standard deviations away from the mean per inference method.

        """

        filtered_protein_objects = {
            x: self.datastore_dict[x].get_protein_objects(
                fdr_restricted=True, false_discovery_rate=false_discovery_rate
            )
            for x in self.datastore_dict.keys()
        }
        number_passing_proteins = {x: len(filtered_protein_objects[x]) for x in filtered_protein_objects.keys()}

        # Calculate how similar the number of passing proteins is for each method
        all_values = [x for x in number_passing_proteins.values()]
        mean = numpy.mean(all_values)
        standard_deviation = statistics.stdev(all_values)
        number_stdev_from_mean_dict = {}
        for key in number_passing_proteins.keys():
            cur_value = number_passing_proteins[key]
            number_stdev_from_mean_dict[key] = (cur_value - mean) / standard_deviation

        return number_stdev_from_mean_dict
