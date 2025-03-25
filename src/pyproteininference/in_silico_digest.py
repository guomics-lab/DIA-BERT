import logging
import re
import sys

from pyteomics import fasta, parser

logger = logging.getLogger(__name__)

# set up our logger
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Digest(object):
    """
    The following class handles data storage of in silico digest data from a fasta formatted sequence database.

    Attributes:
        peptide_to_protein_dictionary (dict): Dictionary of peptides (keys) to protein sets (values).
        protein_to_peptide_dictionary (dict): Dictionary of proteins (keys) to peptide sets (values).
        swiss_prot_protein_set (set): Set of reviewed proteins if they are able to be distinguished from unreviewed
            proteins.
        database_path (str): Path to fasta database file to digest.
        missed_cleavages (int): The number of missed cleavages to allow.
        id_splitting (bool): True/False on whether or not to split a given regex off identifiers.
            This is used to split of "sp|" and "tr|"
            from the database protein strings as sometimes the database will contain those strings while
            the input data will have the strings split already.
            Advanced usage only.
        reviewed_identifier_symbol (str/None): Identifier that distinguishes reviewed from unreviewed proteins.
            Typically this is "sp|". Can also be None type.
        digest_type (str): can be any value in `LIST_OF_DIGEST_TYPES`.
        max_peptide_length (int): Max peptide length to keep for analysis.

    """

    TRYPSIN = "trypsin"
    LYSC = "lysc"
    LIST_OF_DIGEST_TYPES = set(parser.expasy_rules.keys())

    AA_LIST = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "E",
        "Q",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    UNIPROT_STRS = "sp\||tr\|"  # noqa W605
    UNIPROT_STR_REGEX = re.compile(UNIPROT_STRS)
    SP_STRING = "sp|"
    METHIONINE = "M"
    ANY_AMINO_ACID = "X"

    def __init__(self):
        pass


class PyteomicsDigest(Digest):
    """
    This class represents a pyteomics implementation of an in silico digest.
    """

    def __init__(
        self,
        database_path,
        digest_type,
        missed_cleavages,
        reviewed_identifier_symbol,
        max_peptide_length,
        id_splitting=True,
    ):
        """
        The following class creates protein to peptide, peptide to protein, and reviewed protein mappings.

        The input is a fasta database, a protein inference parameter object, and whether or not to split IDs.

        This class sets important attributes for the Digest object such as: `peptide_to_protein_dictionary`,
        `protein_to_peptide_dictionary`, and `swiss_prot_protein_set`.

        Args:
            database_path (str): Path to fasta database file to digest.
            digest_type (str): Must be a value in `LIST_OF_DIGEST_TYPES`.
            missed_cleavages (int): Integer that indicates the maximum number of allowable missed cleavages from
                the ms search.
            reviewed_identifier_symbol (str/None): Symbol that indicates a reviewed identifier.
                If using Uniprot this is typically 'sp|'.
            max_peptide_length (int): The maximum length of peptides to keep for the analysis.
            id_splitting (bool): True/False on whether or not to split a given regex off identifiers.
                This is used to split of "sp|" and "tr|"
                from the database protein strings as sometimes the database will contain those
                strings while the input data will have the strings split already.
                Advanced usage only.

        Example:
            >>> digest = pyproteininference.in_silico_digest.PyteomicsDigest(
            >>>     database_path=database_file,
            >>>     digest_type='trypsin',
            >>>     missed_cleavages=2,
            >>>     reviewed_identifier_symbol='sp|',
            >>>     max_peptide_length=7,
            >>>     id_splitting=False,
            >>> )
        """
        self.peptide_to_protein_dictionary = {}
        self.protein_to_peptide_dictionary = {}
        self.swiss_prot_protein_set = set()
        self.database_path = database_path
        self.missed_cleavages = missed_cleavages
        self.id_splitting = id_splitting
        self.reviewed_identifier_symbol = reviewed_identifier_symbol
        if digest_type in self.LIST_OF_DIGEST_TYPES:
            self.digest_type = digest_type
        else:
            raise ValueError(
                "digest_type must be equal to one of the following {}".format(str(self.LIST_OF_DIGEST_TYPES))
            )
        self.max_peptide_length = max_peptide_length

    def digest_fasta_database(self):
        """
        This method reads in and prepares the fasta database for database digestion and assigns
        the several attributes for the Digest object: `peptide_to_protein_dictionary`,
        `protein_to_peptide_dictionary`, and `swiss_prot_protein_set`.

        Returns:
            None:

        Example:
            >>> digest = pyproteininference.in_silico_digest.PyteomicsDigest(
            >>>     database_path=database_file,
            >>>     digest_type='trypsin',
            >>>     missed_cleavages=2,
            >>>     reviewed_identifier_symbol='sp|',
            >>>     max_peptide_length=7,
            >>>     id_splitting=False,
            >>> )
            >>> digest.digest_fasta_database()

        """
        logger.info("Starting Pyteomics Digest...")
        pep_dict = {}
        prot_dict = {}
        sp_set = set()

        for description, sequence in fasta.read(self.database_path):
            new_peptides = parser.cleave(
                sequence,
                parser.expasy_rules[self.digest_type],
                self.missed_cleavages,
                min_length=self.max_peptide_length,
            )

            # Hopefully this splitting works...
            # IDK how robust this is...
            identifier = description.split(" ")[0]

            # Handle ID Splitting...
            if self.id_splitting:
                identifier_stripped = self.UNIPROT_STR_REGEX.sub("", identifier)
            else:
                identifier_stripped = identifier

            # If reviewed add to sp_set
            if self.reviewed_identifier_symbol:
                if identifier.startswith(self.reviewed_identifier_symbol):
                    sp_set.add(identifier_stripped)

            prot_dict[identifier_stripped] = new_peptides
            met_cleaved_peps = set()
            for peptide in new_peptides:
                pep_dict.setdefault(peptide, set()).add(identifier_stripped)
                # Need to account for potential N-term Methionine Cleavage
                if sequence.startswith(peptide) and peptide.startswith(self.METHIONINE):
                    # If our sequence starts with the current peptide... and our current peptide starts with methionine
                    # Then we remove the methionine from the peptide and add it to our dicts...
                    methionine_cleaved_peptide = peptide[1:]
                    met_cleaved_peps.add(methionine_cleaved_peptide)
            for met_peps in met_cleaved_peps:
                pep_dict.setdefault(met_peps, set()).add(identifier_stripped)
                prot_dict[identifier_stripped].add(met_peps)

        self.swiss_prot_protein_set = sp_set
        self.peptide_to_protein_dictionary = pep_dict
        self.protein_to_peptide_dictionary = prot_dict

        logger.info("Pyteomics Digest Finished...")
