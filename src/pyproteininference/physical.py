import re


class Protein(object):
    """
    The following class is a representation of a Protein that stores characteristics/attributes of a protein for the
        entire analysis.
    We use __slots__ to predefine the attributes the Protein Object can have.
    This is done to speed up runtime of the PI algorithm.

    Attributes:
        identifier (str): String identifier for the Protein object.
        score (float): Float that represents the protein score as output from
            [Score object][pyproteininference.scoring.Score] methods.
        psms (list): List of [Psm][pyproteininference.physical.Psm] objects.
        group_identification (set): Set of group Identifiers that the protein belongs to (int).
        reviewed (bool): True/False on if the identifier is reviewed.
        unreviewed (bool): True/False on if the identifier is reviewed.
        peptides (list): List of non flanking peptide sequences.
        peptide_scores (list): List of Psm scores associated with the protein.
        picked (bool): True/False if the protein passes the picker algo. True if passes. False if does not pass.
        num_peptides (int): Number of peptides that map to the given Protein.
        unique_peptides (list): List of peptide strings that are unique to this protein across the analysis.
        num_unique_peptides (int): Number of unique peptides.
        raw_peptides (list): List of raw peptides. Includes flanking AA and Mods.

    """

    __slots__ = (
        "identifier",
        "score",
        "psms",
        "group_identification",
        "reviewed",
        "unreviewed",
        "peptides",
        "peptide_scores",
        "picked",
        "num_peptides",
        "unique_peptides",
        "num_unique_peptides",
        "raw_peptides",
    )

    def __init__(self, identifier):
        """
        Initialization method for Protein object.

        Args:
            identifier (str): String identifier for the Protein object.

        Example:
            >>> protein = pyproteininference.physical.Protein(identifier = "PRKDC_HUMAN|P78527")

        """
        self.identifier = identifier
        self.score = None
        self.psms = []  # List of psm objects
        self.group_identification = set()
        self.reviewed = False
        self.unreviewed = False
        self.peptides = None  # Sequence info without flanking
        self.peptide_scores = None  # remove
        self.picked = True
        self.num_peptides = None  # remove
        self.unique_peptides = None  # remove
        self.num_unique_peptides = None  # remove
        self.raw_peptides = set()  # Includes Flanking Seq Info

    def get_psm_scores(self):
        """
        Retrieves psm scores for a given protein.

        Returns:
            list: List of psm scores for the given protein.

        """
        score_list = [x.main_score for x in self.psms]
        return score_list

    def get_psm_identifiers(self):
        """
        Retrieves a list of Psm identifiers.

         Returns:
             list: List of Psm identifiers.

        """
        psms = [x.identifier for x in self.psms]
        return psms

    def get_stripped_psm_identifiers(self):
        """
        Retrieves a list of Psm identifiers that have had mods removed and flanking AAs removed.

         Returns:
             list: List of Psm identifiers that have no mods or flanking AAs.

        """
        psms = [x.stripped_peptide for x in self.psms]
        return psms

    def get_unique_peptide_identifiers(self):
        """
        Retrieves the unique set of peptides for a protein.

         Returns:
             set: Set of peptide strings.

        """
        unique_peptides = set(self.get_psm_identifiers())
        return unique_peptides

    def get_unique_stripped_peptide_identifiers(self):
        """
        Retrieves the unique set of peptides for a protein that are stripped.

         Returns:
             set: Set of peptide strings that are stripped of mods and flanking AAs.

        """
        stripped_peptide_identifiers = set(self.get_stripped_psm_identifiers())
        return stripped_peptide_identifiers

    def get_num_psms(self):
        """
        Retrieves the number of Psms.

         Returns:
             int: Number of Psms.

        """
        num_psms = len(self.get_psm_identifiers())
        return num_psms

    def get_num_peptides(self):
        """
        Retrieves the number of peptides.

         Returns:
             int: Number of peptides.

        """
        num_peptides = len(self.get_unique_peptide_identifiers())
        return num_peptides

    def get_psm_ids(self):
        """
        Retrieves the Psm Ids.

         Returns:
            list: List of Psm Ids.

        """
        psm_ids = [x.psm_id for x in self.psms]
        return psm_ids


class Psm(object):
    """
    The following class is a physical Psm class that stores characteristics of a psm for the entire analysis.
    We use __slots__ to predefine the attributes the Psm Object can have.
    This is done to speed up runtime of the PI algorithm.

    Attributes:
        identifier (str): Peptide Identifier: IE "K.DLIDEGH#AATQLVNQLHDVVVENNLSDK.Q".
        percscore (float): Percolator Score from input file if it exists.
        qvalue (float): Q value from input file if it exists.
        pepvalue (float): Pep value from input file if it exists.
        possible_proteins (list): List of protein strings that the Psm maps to based on the digest.
        psm_id (str): String that represents a global identifier for the Psm. Should come from input files.
        custom_score (float): Score that comes from a custom column in the input files.
        main_score (float): The Psm score to be used as the scoring variable for protein scoring. can be
            percscore,qvalue,pepvalue, or custom_score.
        stripped_peptide (str): This is the identifier attribute that has had mods removed and flanking AAs
            removed IE: DLIDEGHAATQLVNQLHDVVVENNLSDK.
        non_flanking_peptide (str): This is the identifier attribute that has had flanking AAs
            removed IE: DLIDEGH#AATQLVNQLHDVVVENNLSDK. #NOTE Mods are still present here.

    """

    __slots__ = (
        "identifier",
        "percscore",
        "qvalue",
        "pepvalue",
        "possible_proteins",
        "psm_id",
        "custom_score",
        "main_score",
        "stripped_peptide",
        "non_flanking_peptide",
    )

    # The regex removes anything between parantheses including parenthases - \([^()]*\)
    # The regex removes anything between brackets including parenthases - \[.*?\]
    # And the regex removes anything that is not an A-Z character [^A-Z]
    MOD_REGEX = re.compile("\([^()]*\)|\[.*?\]|[^A-Z]")  # noqa W605

    FRONT_FLANKING_REGEX = re.compile("^[A-Z|-][.]")
    BACK_FLANKING_REGEX = re.compile("[.][A-Z|-]$")

    SCORE_ATTRIBUTE_NAMES = set(["pepvalue", "qvalue", "percscore", "custom_score"])

    def __init__(self, identifier):
        """
        Initialization method for the Psm object.
        This method also initializes the `stripped_peptide` and `non_flanking_peptide` attributes.

        Args:
            identifier (str): Peptide Identifier: IE ""K.DLIDEGH#AATQLVNQLHDVVVENNLSDK.Q".

        Example:
            >>> psm = pyproteininference.physical.Psm(identifier = "K.DLIDEGHAATQLVNQLHDVVVENNLSDK.Q")

        """
        self.identifier = identifier
        self.percscore = None
        self.qvalue = None
        self.pepvalue = None
        self.possible_proteins = None
        self.psm_id = None
        self.custom_score = None
        self.main_score = None
        self.stripped_peptide = None
        self.non_flanking_peptide = None

        # Add logic to split the peptide and strip it of mods
        current_peptide = Psm.split_peptide(peptide_string=self.identifier)

        self.non_flanking_peptide = current_peptide

        if not current_peptide.isupper() or not current_peptide.isalpha():
            # If we have mods remove them...
            peptide_string = current_peptide.upper()
            stripped_peptide = Psm.remove_peptide_mods(peptide_string)
            current_peptide = stripped_peptide

        # Set stripped_peptide variable
        self.stripped_peptide = current_peptide

    @classmethod
    def remove_peptide_mods(cls, peptide_string):
        """
        This class method takes a string and uses a `MOD_REGEX` to remove mods from peptide strings.

        Args:
            peptide_string (str): Peptide string to have mods removed from.

        Returns:
            str: a peptide string with mods removed.

        """
        stripped_peptide = cls.MOD_REGEX.sub("", peptide_string)
        return stripped_peptide

    @classmethod
    def split_peptide(cls, peptide_string, delimiter="."):
        """
        This class method takes a peptide string with flanking AAs and removes them from the peptide string.
        This method uses string splitting and if the method produces a faulty peptide the method
            [split_peptide_pro][pyproteininference.physical.Psm.split_peptide_pro] will be called.

        Args:
            peptide_string (str): Peptide string to have mods removed from.
            delimiter (str): a string to indicate what separates a leading/trailing (flanking) AA from the
                peptide sequence.

        Returns:
            str: a peptide string with flanking AAs removed.

        """
        peptide_split = peptide_string.split(delimiter)
        if len(peptide_split) == 3:
            # If we get 3 chunks it will usually be ['A', 'ADGSDFGSS', 'F']
            # So take index 1
            peptide = peptide_split[1]
        elif len(peptide_split) == 1:
            # If we get 1 chunk it should just be ['ADGSDFGSS']
            # So take index 0
            peptide = peptide_split[0]
        else:
            # If we split the peptide and it is not length 1 or 3 then try to split with pro
            peptide = cls.split_peptide_pro(peptide_string=peptide_string, delimiter=delimiter)

        return peptide

    @classmethod
    def split_peptide_pro(cls, peptide_string, delimiter="."):
        """
        This class method takes a peptide string with flanking AAs and removes them from the peptide string.
        This is a specialized method of [split_peptide][pyproteininference.physical.Psm.split_peptide] that uses
         regex identifiers to replace flanking AAs as opposed to string splitting.


        Args:
            peptide_string (str): Peptide string to have mods removed from.
            delimiter (str): a string to indicate what separates a leading/trailing (flanking) AA from the peptide
                sequence.

        Returns:
            str: a peptide string with flanking AAs removed.

        """

        if delimiter != ".":
            front_regex = "^[A-Z|-][{}]".format(delimiter)
            cls.FRONT_FLANKING_REGEX = re.compile(front_regex)
            back_regex = "[{}][A-Z|-]$".format(delimiter)
            cls.BACK_FLANKING_REGEX = re.compile(back_regex)

        # Replace the front flanking with nothing
        peptide_string = cls.FRONT_FLANKING_REGEX.sub("", peptide_string)

        # Replace the back flanking with nothing
        peptide_string = cls.BACK_FLANKING_REGEX.sub("", peptide_string)

        return peptide_string

    def assign_main_score(self, score):
        """
        This method takes in a score type and assigns the variable main_score for a given Psm based on the score type.

        Args:
            score (str): This is a string representation of the Psm attribute that will get assigned to the main_score
                variable.

        """
        # Assign a main score based on user input
        if score not in self.SCORE_ATTRIBUTE_NAMES:
            raise ValueError("Scores must either be one of: '{}'".format(", ".join(self.SCORE_ATTRIBUTE_NAMES)))
        else:
            self.main_score = getattr(self, score)


class ProteinGroup(object):
    """
    The following class is a physical Protein Group class that stores characteristics of a Protein Group for the entire
        analysis.
    We use __slots__ to predefine the attributes the Psm Object can have.
    This is done to speed up runtime of the PI algorithm.

    Attributes:
        number_id (int): unique Integer to represent a group.
        proteins (list): List of [Protein][pyproteininference.physical.Protein] objects.
        q_value (float): Q value for the protein group that is calculated with method
            [calculate_q_values][pyproteininference.datastore.DataStore.calculate_q_values].

    """

    __slots__ = ("proteins", "number_id", "q_value")

    def __init__(self, number_id):
        """
        Initialization method for ProteinGroup object.

        Args:
            number_id (int): unique Integer to represent a group.

        Example:
            >>> pg = pyproteininference.physical.ProteinGroup(number_id = 1)
        """

        self.proteins = []
        self.number_id = number_id
        self.q_value = None
