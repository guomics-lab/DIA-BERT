import logging
import os
import sys

from src.pyproteininference import parameters
from src.pyproteininference import in_silico_digest
from src.pyproteininference import reader as pi_reader
from src.pyproteininference import datastore as pi_datastore
from src.pyproteininference import scoring as pi_scoring
from src.pyproteininference import export as pi_export
from src.pyproteininference import inference as pi_inference
from src.pyproteininference.inference import Inference

logger = logging.getLogger(__name__)

# set up our logger
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class ProteinInferencePipeline(object):
    """
    This is the main Protein Inference class which houses the logic of the entire data analysis pipeline.
    Logic is executed in the [execute][pyproteininference.pipeline.ProteinInferencePipeline.execute] method.

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
        id_splitting (bool): True/False on whether to split protein IDs in the digest. Advanced usage only.
        append_alt_from_db (bool): True/False on whether to append alternative proteins from the DB digestion in
            Reader class.
        data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
        digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].

    """

    def __init__(
        self,
        parameter_file,
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
            output_filename (str): Path to Filename where output will be written. Will override output_directory.
            output_directory (str): Path to Directory where output will be written.
            id_splitting (bool): True/False on whether to split protein IDs in the digest. Advanced usage only.
            append_alt_from_db (bool): True/False on whether to append alternative proteins from the DB digestion in
                Reader class.

        Returns:
            object:

        Example:
            >>> pipeline = pyproteininference.pipeline.ProteinInferencePipeline(
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
        self.data = None
        self.digest = None

        self._validate_input()

        self._set_output_directory()

        self._log_append_alt_from_db()

        self._log_id_splitting()

    def execute(self):
        """
        This method is the main driver of the data analysis for the protein inference package.
        This method calls other classes and methods that make up the protein inference pipeline.
        This includes but is not limited to:

        This method sets the data [DataStore Object][pyproteininference.datastore.DataStore] and digest
            [Digest Object][pyproteininference.in_silico_digest.Digest].

        1. Parameter file management.
        2. Digesting Fasta Database (Optional).
        3. Reading in input Psm Files.
        4. Initializing the [DataStore Object][pyproteininference.datastore.DataStore].
        5. Restricting Psms.
        6. Creating Protein objects/scoring input.
        7. Scoring Proteins.
        8. Running Protein Picker.
        9. Running Inference Methods/Grouping.
        10. Calculating Q Values.
        11. Exporting Proteins to filesystem.

        Example:
            >>> pipeline = pyproteininference.pipeline.ProteinInferencePipeline(
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
            >>> )
            >>> pipeline.execute()

        """
        # STEP 1: Load parameter file #
        # STEP 1: Load parameter file #
        # STEP 1: Load parameter file #
        pyproteininference_parameters = parameters.ProteinInferenceParameter(
            yaml_param_filepath=self.parameter_file
        )

        # STEP 2: Start with running an In Silico Digestion #
        # STEP 2: Start with running an In Silico Digestion #
        # STEP 2: Start with running an In Silico Digestion #
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

        # STEP 3: Read PSM Data #
        # STEP 3: Read PSM Data #
        # STEP 3: Read PSM Data #
        reader = pi_reader.GenericReader(
            target_file=self.target_files,
            decoy_file=self.decoy_files,
            combined_files=self.combined_files,
            parameter_file_object=pyproteininference_parameters,
            digest=digest,
            append_alt_from_db=self.append_alt_from_db,
        )
        reader.read_psms()

        # STEP 4: Initiate the datastore object #
        # STEP 4: Initiate the datastore object #
        # STEP 4: Initiate the datastore object #
        data = pi_datastore.DataStore(reader=reader, digest=digest)

        # Step 5: Restrict the PSM data
        # Step 5: Restrict the PSM data
        # Step 5: Restrict the PSM data
        data.restrict_psm_data()

        data.recover_mapping()
        # Step 6: Generate protein scoring input
        # Step 6: Generate protein scoring input
        # Step 6: Generate protein scoring input
        data.create_scoring_input()

        # Step 7: Remove non unique peptides if running exclusion
        # Step 7: Remove non unique peptides if running exclusion
        # Step 7: Remove non unique peptides if running exclusion
        if pyproteininference_parameters.inference_type == Inference.EXCLUSION:
            # This gets ran if we run exclusion...
            data.exclude_non_distinguishing_peptides()

        # STEP 8: Score our PSMs given a score method
        # STEP 8: Score our PSMs given a score method
        # STEP 8: Score our PSMs given a score method
        score = pi_scoring.Score(data=data)
        score.score_psms(score_method=pyproteininference_parameters.protein_score)

        # STEP 9: Run protein picker on the data
        # STEP 9: Run protein picker on the data
        # STEP 9: Run protein picker on the data
        if pyproteininference_parameters.picker:
            data.protein_picker()
        else:
            pass

        # STEP 10: Apply Inference
        # STEP 10: Apply Inference
        # STEP 10: Apply Inference
        pi_inference.Inference.run_inference(data=data, digest=digest)

        # STEP 11: Q value Calculations
        # STEP 11: Q value Calculations
        # STEP 11: Q value Calculations
        data.calculate_q_values()

        # STEP 12: Export to CSV
        # STEP 12: Export to CSV
        # STEP 12: Export to CSV
        export = pi_export.Export(data=data)
        export.export_to_csv(
            output_filename=self.output_filename,
            directory=self.output_directory,
            export_type=pyproteininference_parameters.export,
        )

        self.data = data
        self.digest = digest

        logger.info("Protein Inference Finished")

    def _validate_input(self):
        """
        Internal method that validates whether the proper input files have been defined.

        One of the following combinations must be selected as input. No more and no less:

        1. either one or multiple target_files and decoy_files.
        2. either one or multiple combined_files that include target and decoy data.
        3. a directory that contains target files (target_directory) as well as a directory that contains decoy files
            (decoy_directory).
        4. a directory that contains combined target/decoy files (combined_directory).

        Raises:
            ValueError: ValueError will occur if an improper combination of input.
        """
        if (
            self.target_files
            and self.decoy_files
            and not self.combined_files
            and not self.target_directory
            and not self.decoy_directory
            and not self.combined_directory
        ):
            logger.info("Validating input as target_files and decoy_files")
        elif (
            self.combined_files
            and not self.target_files
            and not self.decoy_files
            and not self.decoy_directory
            and not self.target_directory
            and not self.combined_directory
        ):
            logger.info("Validating input as combined_files")
        elif (
            self.target_directory
            and self.decoy_directory
            and not self.target_files
            and not self.decoy_files
            and not self.combined_directory
            and not self.combined_files
        ):
            logger.info("Validating input as target_directory and decoy_directory")
            self._transform_directory_to_files()
        elif (
            self.combined_directory
            and not self.combined_files
            and not self.decoy_files
            and not self.decoy_directory
            and not self.target_files
            and not self.target_directory
        ):
            logger.info("Validating input as combined_directory")
            self._transform_directory_to_files()
        else:
            raise ValueError(
                "To run Protein inference please supply either: "
                "(1) either one or multiple target_files and decoy_files, "
                "(2) either one or multiple combined_files that include target and decoy data"
                "(3) a directory that contains target files (target_directory) as well as a directory that "
                "contains decoy files (decoy_directory)"
                "(4) a directory that contains combined target/decoy files (combined_directory)"
            )

    def _transform_directory_to_files(self):
        """
        This internal method takes files that are in the target_directory, decoy_directory, or combined_directory and
        reassigns these files to the target_files, decoy_files, and combined_files to be used in
         [Reader][pyproteininference.reader.Reader] object.
        """
        if self.target_directory and self.decoy_directory:
            logger.info("Transforming target_directory and decoy_directory into files")
            target_files = os.listdir(self.target_directory)
            target_files_full = [
                os.path.join(self.target_directory, x) for x in target_files if x.endswith(".txt") or x.endswith(".tsv")
            ]

            decoy_files = os.listdir(self.decoy_directory)
            decoy_files_full = [
                os.path.join(self.decoy_directory, x) for x in decoy_files if x.endswith(".txt") or x.endswith(".tsv")
            ]

            self.target_files = target_files_full
            self.decoy_files = decoy_files_full

        elif self.combined_directory:
            logger.info("Transforming combined_directory into files")
            combined_files = os.listdir(self.combined_directory)
            combined_files_full = [
                os.path.join(self.combined_directory, x)
                for x in combined_files
                if x.endswith(".txt") or x.endswith(".tsv")
            ]
            self.combined_files = combined_files_full

    def _set_output_directory(self):
        """
        Internal method for setting the output directory.
        If the output_directory argument is not supplied the output directory is set as the cwd.
        """
        if not self.output_directory:
            self.output_directory = os.getcwd()
        else:
            pass

    def _log_append_alt_from_db(self):
        """
        Internal method for logging whether the user sets alternative protein append to True or False.
        """
        if self.append_alt_from_db:
            logger.info("Append Alternative Proteins from Database set to True")
        else:
            logger.info("Append Alternative Proteins from Database set to False")

    def _log_id_splitting(self):
        """
        Internal method for logging whether the user sets ID splitting to True or False.
        """
        if self.id_splitting:
            logger.info("ID Splitting for Database Digestion set to True")
        else:
            logger.info("ID Splitting for Database Digestion set to False")
