import csv
import logging
import os
import sys

logger = logging.getLogger(__name__)

# set up our logger
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Export(object):
    """
    Class that handles exporting protein inference results to filesystem as csv files.

    Attributes:
        data (DataStore): [DataStore object][pyproteininference.datastore.DataStore].
        filepath (str): Path to file to be written.

    """

    EXPORT_LEADS = "leads"
    EXPORT_ALL = "all"
    EXPORT_COMMA_SEP = "comma_sep"
    EXPORT_Q_VALUE_COMMA_SEP = "q_value_comma_sep"
    EXPORT_Q_VALUE = "q_value"
    EXPORT_Q_VALUE_ALL = "q_value_all"
    EXPORT_PEPTIDES = "peptides"
    EXPORT_PSMS = "psms"
    EXPORT_PSM_IDS = "psm_ids"
    EXPORT_LONG = "long"

    EXPORT_TYPES = [
        EXPORT_LEADS,
        EXPORT_ALL,
        EXPORT_COMMA_SEP,
        EXPORT_Q_VALUE_COMMA_SEP,
        EXPORT_Q_VALUE,
        EXPORT_Q_VALUE_ALL,
        EXPORT_PEPTIDES,
        EXPORT_PSMS,
        EXPORT_PSM_IDS,
        EXPORT_LONG,
    ]

    def __init__(self, data):
        """
        Initialization method for the Export class.

        Args:
            data (DataStore): [DataStore object][pyproteininference.datastore.DataStore].

        Example:
            >>> export = pyproteininference.export.Export(data=data)

        """
        self.data = data
        self.filepath = None

    def export_to_csv(self, output_filename=None, directory=None, export_type="q_value"):
        """
        Method that dispatches to one of the many export methods given an export_type input.

        filepath is determined based on directory arg and information from
        [DataStore object][pyproteininference.datastore.DataStore].

        This method sets the `filepath` variable.

        Args:
            output_filename (str): Filepath to write to. If set as None will auto generate filename and
                will write to directory variable.
            directory (str): Directory to write the result file to. If None, will write to current working directory.
            export_type (str): Must be a value in `EXPORT_TYPES` and determines the output format.

        Example:
            >>> export = pyproteininference.export.Export(data=data)
            >>> export.export_to_csv(output_filename=None, directory="/path/to/output/dir/", export_type="psms")

        """

        if not directory:
            directory = os.getcwd()

        data = self.data
        tag = data.parameter_file_object.tag

        if self.EXPORT_LEADS == export_type:
            filename = "{}_leads_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_leads_restricted(filename_out=complete_filepath)

        elif self.EXPORT_ALL == export_type:
            filename = "{}_all_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_all_restricted(complete_filepath)

        elif self.EXPORT_COMMA_SEP == export_type:
            filename = "{}_comma_sep_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_comma_sep_restricted(complete_filepath)

        elif self.EXPORT_Q_VALUE_COMMA_SEP == export_type:
            filename = "{}_q_value_comma_sep_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_q_value_comma_sep(complete_filepath)

        elif self.EXPORT_Q_VALUE == export_type:
            filename = "{}_q_value_leads_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_q_value_leads(complete_filepath)

        elif self.EXPORT_Q_VALUE_ALL == export_type:
            filename = "{}_q_value_all_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_q_value_all(complete_filepath)

        elif self.EXPORT_PEPTIDES == export_type:
            filename = "{}_q_value_leads_peptides_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_q_value_leads_peptides(complete_filepath)

        elif self.EXPORT_PSMS == export_type:
            filename = "{}_q_value_leads_psms_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_q_value_leads_psms(complete_filepath)

        elif self.EXPORT_PSM_IDS == export_type:
            filename = "{}_q_value_leads_psm_ids_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_q_value_leads_psm_ids(complete_filepath)

        elif self.EXPORT_LONG == export_type:
            filename = "{}_q_value_long_{}_{}.csv".format(tag, data.short_protein_score, data.psm_score)
            complete_filepath = os.path.join(directory, filename)
            if output_filename:
                complete_filepath = output_filename
            logger.info("Exporting Protein Inference Data to File: {}".format(complete_filepath))
            self.csv_export_q_value_leads_long(complete_filepath)

        else:
            complete_filepath = "protein_inference_results.csv"

        self.filepath = complete_filepath

    def csv_export_all_restricted(self, filename_out):
        """
        Method that outputs a subset of the passing proteins based on FDR.

        This method returns a non-square CSV file.

        Args:
            filename_out (str): Filename for the data to be written to

        """
        protein_objects = self.data.get_protein_objects(fdr_restricted=True)
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in protein_objects:
            for prots in groups:
                protein_export_list.append([prots.identifier])
                protein_export_list[-1].append(prots.score)
                protein_export_list[-1].append(prots.num_peptides)
                if prots.reviewed:
                    protein_export_list[-1].append("Reviewed")
                else:
                    protein_export_list[-1].append("Unreviewed")
                protein_export_list[-1].append(prots.group_identification)
                for peps in prots.peptides:
                    protein_export_list[-1].append(peps)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_leads_restricted(self, filename_out):
        """
        Method that outputs a subset of the passing proteins based on FDR.
        Only Proteins that pass FDR will be output and only Lead proteins will be output

        This method returns a non-square CSV file.

        Args:
            filename_out (str): Filename for the data to be written to.

        """
        protein_objects = self.data.get_protein_objects(fdr_restricted=True)
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in protein_objects:
            protein_export_list.append([groups[0].identifier])
            protein_export_list[-1].append(groups[0].score)
            protein_export_list[-1].append(groups[0].num_peptides)
            if groups[0].reviewed:
                protein_export_list[-1].append("Reviewed")
            else:
                protein_export_list[-1].append("Unreviewed")
            protein_export_list[-1].append(groups[0].group_identification)
            for peps in sorted(groups[0].peptides):
                protein_export_list[-1].append(peps)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_comma_sep_restricted(self, filename_out):
        """
        Method that outputs a subset of the passing proteins based on FDR.
        Only Proteins that pass FDR will be output and only Lead proteins will be output.
        Proteins in the groups of lead proteins will also be output in the same row as the lead.

        This method returns a non-square CSV file.

        Args:
            filename_out (str): Filename for the data to be written to.

        """
        protein_objects = self.data.get_protein_objects(fdr_restricted=True)
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Other_Potential_Identifiers",
            ]
        ]
        for groups in protein_objects:
            for prots in groups:
                if prots == groups[0]:
                    protein_export_list.append([prots.identifier])
                    protein_export_list[-1].append(prots.score)
                    protein_export_list[-1].append(prots.num_peptides)
                    if prots.reviewed:
                        protein_export_list[-1].append("Reviewed")
                    else:
                        protein_export_list[-1].append("Unreviewed")
                    protein_export_list[-1].append(prots.group_identification)
                else:
                    protein_export_list[-1].append(prots.identifier)
        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_q_value_leads(self, filename_out):
        """
        Method that outputs all lead proteins with Q values.

        This method returns a non-square CSV file.

        Args:
            filename_out (str): Filename for the data to be written to.

        """
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Q_Value",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in self.data.protein_group_objects:
            lead_protein = groups.proteins[0]
            protein_export_list.append([lead_protein.identifier])
            protein_export_list[-1].append(lead_protein.score)
            protein_export_list[-1].append(groups.q_value)
            protein_export_list[-1].append(lead_protein.num_peptides)
            if lead_protein.reviewed:
                protein_export_list[-1].append("Reviewed")
            else:
                protein_export_list[-1].append("Unreviewed")
            protein_export_list[-1].append(groups.number_id)
            peptides = lead_protein.peptides
            for peps in sorted(peptides):
                protein_export_list[-1].append(peps)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_q_value_comma_sep(self, filename_out):
        """
        Method that outputs all lead proteins with Q values.
        Proteins in the groups of lead proteins will also be output in the same row as the lead.

        This method returns a non-square CSV file.

        Args:
            filename_out (str): Filename for the data to be written to.

        """
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Q_Value",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Other_Potential_Identifiers",
            ]
        ]
        for groups in self.data.protein_group_objects:
            lead_protein = groups.proteins[0]
            protein_export_list.append([lead_protein.identifier])
            protein_export_list[-1].append(lead_protein.score)
            protein_export_list[-1].append(groups.q_value)
            protein_export_list[-1].append(lead_protein.num_peptides)
            if lead_protein.reviewed:
                protein_export_list[-1].append("Reviewed")
            else:
                protein_export_list[-1].append("Unreviewed")
            protein_export_list[-1].append(groups.number_id)
            for other_prots in groups.proteins[1:]:
                protein_export_list[-1].append(other_prots.identifier)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_q_value_all(self, filename_out):
        """
        Method that outputs all proteins with Q values.
        Non Lead proteins are also output - entire group gets output.
        Proteins in the groups of lead proteins will also be output in the same row as the lead.

        This method returns a non-square CSV file.

        Args:
            filename_out (str): Filename for the data to be written to.

        """
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Q_Value",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in self.data.protein_group_objects:
            for proteins in groups.proteins:
                protein_export_list.append([proteins.identifier])
                protein_export_list[-1].append(proteins.score)
                protein_export_list[-1].append(groups.q_value)
                protein_export_list[-1].append(proteins.num_peptides)
                if proteins.reviewed:
                    protein_export_list[-1].append("Reviewed")
                else:
                    protein_export_list[-1].append("Unreviewed")
                protein_export_list[-1].append(groups.number_id)
                for peps in sorted(proteins.peptides):
                    protein_export_list[-1].append(peps)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_q_value_all_proteologic(self, filename_out):
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Q_Value",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in self.data.protein_group_objects:
            for proteins in groups.proteins:
                protein_export_list.append([proteins.identifier])
                protein_export_list[-1].append(proteins.score)
                protein_export_list[-1].append(groups.q_value)
                protein_export_list[-1].append(proteins.num_peptides)
                if proteins.reviewed:
                    protein_export_list[-1].append("Reviewed")
                else:
                    protein_export_list[-1].append("Unreviewed")
                protein_export_list[-1].append(groups.number_id)
                for peps in sorted(proteins.peptides):
                    protein_export_list[-1].append(peps)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_q_value_leads_long(self, filename_out):
        """
        Method that outputs all lead proteins with Q values.

        This method returns a long formatted result file with one peptide on each row.

        Args:
            filename_out (str): Filename for the data to be written to.

        """
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Q_Value",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in self.data.protein_group_objects:
            lead_protein = groups.proteins[0]
            for peps in sorted(lead_protein.peptides):
                protein_export_list.append([lead_protein.identifier])
                protein_export_list[-1].append(lead_protein.score)
                protein_export_list[-1].append(groups.q_value)
                protein_export_list[-1].append(lead_protein.num_peptides)
                if lead_protein.reviewed:
                    protein_export_list[-1].append("Reviewed")
                else:
                    protein_export_list[-1].append("Unreviewed")
                protein_export_list[-1].append(groups.number_id)
                protein_export_list[-1].append(peps)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_q_value_leads_peptides(self, filename_out, peptide_delimiter=" "):
        """
        Method that outputs all lead proteins with Q values in rectangular format.
        This method outputs unique peptides per protein.

        This method returns a rectangular CSV file.

        Args:
            filename_out (str): Filename for the data to be written to.
            peptide_delimiter (str): String to separate peptides by in the "Peptides" column of the csv file
        """
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Q_Value",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in self.data.protein_group_objects:
            lead_protein = groups.proteins[0]
            protein_export_list.append([lead_protein.identifier])
            protein_export_list[-1].append(lead_protein.score)
            protein_export_list[-1].append(groups.q_value)
            protein_export_list[-1].append(lead_protein.num_peptides)
            if lead_protein.reviewed:
                protein_export_list[-1].append("Reviewed")
            else:
                protein_export_list[-1].append("Unreviewed")
            protein_export_list[-1].append(groups.number_id)
            peptides = peptide_delimiter.join(list(sorted(lead_protein.peptides)))
            protein_export_list[-1].append(peptides)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_q_value_leads_psms(self, filename_out, peptide_delimiter=" "):
        """
        Method that outputs all lead proteins with Q values in rectangular format.
        This method outputs all PSMs for the protein not just unique peptide identifiers.

        This method returns a rectangular CSV file.

        Args:
            filename_out (str): Filename for the data to be written to.
            peptide_delimiter (str): String to separate peptides by in the "Peptides" column of the csv file.
        """
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Q_Value",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in self.data.protein_group_objects:
            lead_protein = groups.proteins[0]
            protein_export_list.append([lead_protein.identifier])
            protein_export_list[-1].append(lead_protein.score)
            protein_export_list[-1].append(groups.q_value)
            protein_export_list[-1].append(lead_protein.num_peptides)
            if lead_protein.reviewed:
                protein_export_list[-1].append("Reviewed")
            else:
                protein_export_list[-1].append("Unreviewed")
            protein_export_list[-1].append(groups.number_id)
            psms = peptide_delimiter.join(sorted([x.non_flanking_peptide for x in lead_protein.psms]))
            protein_export_list[-1].append(psms)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)

    def csv_export_q_value_leads_psm_ids(self, filename_out, peptide_delimiter=" "):
        """
        Method that outputs all lead proteins with Q values in rectangular format.
        Psms are output as the psm_id value. So sequence information is not output.

        This method returns a rectangular CSV file.

        Args:
            filename_out (str): Filename for the data to be written to.
            peptide_delimiter (str): String to separate psm_ids by in the "Peptides" column of the csv file.
        """
        protein_export_list = [
            [
                "Protein",
                "Score",
                "Q_Value",
                "Number_of_Peptides",
                "Identifier_Type",
                "GroupID",
                "Peptides",
            ]
        ]
        for groups in self.data.protein_group_objects:
            lead_protein = groups.proteins[0]
            protein_export_list.append([lead_protein.identifier])
            protein_export_list[-1].append(lead_protein.score)
            protein_export_list[-1].append(groups.q_value)
            protein_export_list[-1].append(lead_protein.num_peptides)
            if lead_protein.reviewed:
                protein_export_list[-1].append("Reviewed")
            else:
                protein_export_list[-1].append("Unreviewed")
            protein_export_list[-1].append(groups.number_id)
            psms = peptide_delimiter.join(sorted(lead_protein.get_psm_ids()))
            protein_export_list[-1].append(psms)

        with open(filename_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(protein_export_list)
