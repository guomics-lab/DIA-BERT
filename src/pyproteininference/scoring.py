import logging
import math
import sys
from functools import reduce

import numpy

logger = logging.getLogger(__name__)

# set up our logger
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Score(object):
    """
    Score class that contains methods to do a variety of scoring methods on the
    [Psm][pyproteininference.physical.Psm] objects
    contained inside of [Protein][pyproteininference.physical.Protein] objects.

    Methods in the class loop over each Protein object and creates a protein "score" variable using the Psm object
    scores.

    Methods score all proteins from `scoring_input` from [DataStore object][pyproteininference.datastore.DataStore].
    The PSM score that is used is determined from
    [create_scoring_input][pyproteininference.datastore.DataStore.create_scoring_input].

    Each scoring method will set the following attributes for
    the [DataStore object][pyproteininference.datastore.DataStore].

    1. `score_method`; This is the full name of the score method.
    2. `short_score_method`; This is the short name of the score method.
    3. `scored_proteins`; This is a list of [Protein][pyproteininference.physical.Protein] objects
    that have been scored.

    Attributes:
        pre_score_data (list): This is a list of [Protein][pyproteininference.physical.Protein] objects
            that contain [Psm][pyproteininference.physical.Psm] objects.
        data (DataStore): [DataStore][pyproteininference.datastore.DataStore] object.

    """

    BEST_PEPTIDE_PER_PROTEIN = "best_peptide_per_protein"
    ITERATIVE_DOWNWEIGHTED_LOG = "iterative_downweighted_log"
    MULTIPLICATIVE_LOG = "multiplicative_log"
    DOWNWEIGHTED_MULTIPLICATIVE_LOG = "downweighted_multiplicative_log"
    DOWNWEIGHTED_VERSION2 = "downweighted_version2"
    TOP_TWO_COMBINED = "top_two_combined"
    GEOMETRIC_MEAN = "geometric_mean"
    ADDITIVE = "additive"

    SCORE_METHODS = [
        BEST_PEPTIDE_PER_PROTEIN,
        ITERATIVE_DOWNWEIGHTED_LOG,
        MULTIPLICATIVE_LOG,
        DOWNWEIGHTED_MULTIPLICATIVE_LOG,
        DOWNWEIGHTED_VERSION2,
        TOP_TWO_COMBINED,
        GEOMETRIC_MEAN,
        ADDITIVE,
    ]

    SHORT_BEST_PEPTIDE_PER_PROTEIN = "bppp"
    SHORT_ITERATIVE_DOWNWEIGHTED_LOG = "idwl"
    SHORT_MULTIPLICATIVE_LOG = "ml"
    SHORT_DOWNWEIGHTED_MULTIPLICATIVE_LOG = "dwml"
    SHORT_DOWNWEIGHTED_VERSION2 = "dw2"
    SHORT_TOP_TWO_COMBINED = "ttc"
    SHORT_GEOMETRIC_MEAN = "gm"
    SHORT_ADDITIVE = "add"

    SHORT_SCORE_METHODS = [
        SHORT_BEST_PEPTIDE_PER_PROTEIN,
        SHORT_ITERATIVE_DOWNWEIGHTED_LOG,
        SHORT_MULTIPLICATIVE_LOG,
        SHORT_DOWNWEIGHTED_MULTIPLICATIVE_LOG,
        SHORT_DOWNWEIGHTED_VERSION2,
        SHORT_TOP_TWO_COMBINED,
        SHORT_GEOMETRIC_MEAN,
        SHORT_ADDITIVE,
    ]

    MULTIPLICATIVE_SCORE_TYPE = "multiplicative"
    ADDITIVE_SCORE_TYPE = "additive"

    SCORE_TYPES = [MULTIPLICATIVE_SCORE_TYPE, ADDITIVE_SCORE_TYPE]

    def __init__(self, data):
        """
        Initialization method for the Score class.

        Args:
            data (DataStore): [DataStore][pyproteininference.datastore.DataStore] object.

        Raises:
            ValueError: If the variable `scoring_input` for the [DataStore][pyproteininference.datastore.DataStore]
                object is Empty "[]" or does not exist "None".

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
        """
        if data.scoring_input:
            self.pre_score_data = data.scoring_input
        else:
            raise ValueError(
                "scoring input not found in data object - Please run 'create_scoring_input' method from "
                "DataStore to run any scoring type"
            )
        self.data = data

    def score_psms(self, score_method="multiplicative_log"):
        """
        This method dispatches to the actual scoring method given a string input that is defined in the
        [ProteinInferenceParameter][pyproteininference.parameters.ProteinInferenceParameter] object.

        Args:
            score_method (str): This is a string that represents which scoring method to call.

        Raises:
            ValueError: Will Error out if the score_method is not present in the constant `SCORE_METHODS`.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.score_psms(score_method="best_peptide_per_protein")
        """

        if score_method not in self.SCORE_METHODS:
            raise ValueError(
                "score method '{}' is not a proper method. Score method must be one of the following: '{}'".format(
                    score_method, ", ".join(self.SCORE_METHODS)
                )
            )
        else:
            if score_method == self.BEST_PEPTIDE_PER_PROTEIN:
                self.best_peptide_per_protein()
            if score_method == self.ITERATIVE_DOWNWEIGHTED_LOG:
                self.iterative_down_weighted_log()
            if score_method == self.MULTIPLICATIVE_LOG:
                self.multiplicative_log()
            if score_method == self.DOWNWEIGHTED_MULTIPLICATIVE_LOG:
                self.down_weighted_multiplicative_log()
            if score_method == self.DOWNWEIGHTED_VERSION2:
                self.down_weighted_v2()
            if score_method == self.TOP_TWO_COMBINED:
                self.top_two_combied()
            if score_method == self.GEOMETRIC_MEAN:
                self.geometric_mean_log()
            if score_method == self.ADDITIVE:
                self.additive()

    def best_peptide_per_protein(self):
        """
        This method uses a best peptide per protein scoring scheme.
        The top scoring Psm for each protein is selected as the overall Protein object score.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.best_peptide_per_protein()

        """

        all_scores = []

        logger.info("Scoring Proteins with BPPP")
        for protein in self.pre_score_data:
            val_list = protein.get_psm_scores()
            score = min([float(x) for x in val_list])

            protein.score = score

            all_scores.append(protein)
        # Here do ascending sorting because a lower pep or q value is better
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=False)

        self.data.protein_score = self.BEST_PEPTIDE_PER_PROTEIN
        self.data.short_protein_score = self.SHORT_BEST_PEPTIDE_PER_PROTEIN
        self.data.scored_proteins = all_scores

    def fishers_method(self):
        """
        This method uses a fishers method scoring scheme.
\
        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.fishers_method()

         """

        all_scores = []
        logger.info("Scoring Proteins with fishers method")
        for protein in self.pre_score_data:
            val_list = protein.get_psm_scores()
            score = -2 * sum([math.log(x) for x in val_list])

            protein.score = score

            all_scores.append(protein)
        # Here reverse the sorting to descending because a higher score is better
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)
        self.data.protein_score = "fishers_method"
        self.data.short_protein_score = "fm"
        self.data.scored_proteins = all_scores

    def multiplicative_log(self):
        """
        This method uses a Multiplicative Log scoring scheme.
        The selected Psm score from all the peptides per protein are multiplied together and we take -Log(X)
        of the multiplied Peptide scores.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.multiplicative_log()
        """

        # Instead of making all_scores a list... make it a generator??

        all_scores = []
        logger.info("Scoring Proteins with Multiplicative Log Method")
        for protein in self.pre_score_data:
            # We create a generator of val_list...
            val_list = protein.get_psm_scores()

            combine = reduce(lambda x, y: x * y, val_list)
            if combine == 0:
                combine = sys.float_info.min
            score = -math.log(combine)
            protein.score = score

            all_scores.append(protein)

        # Higher score is better as a smaller q or pep in a -log will give a larger value
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)

        self.data.protein_score = self.MULTIPLICATIVE_LOG
        self.data.short_protein_score = self.SHORT_MULTIPLICATIVE_LOG
        self.data.scored_proteins = all_scores

    def down_weighted_multiplicative_log(self):
        """
        This method uses a Multiplicative Log scoring scheme.
        The selected PSM score from all the peptides per protein are multiplied together and
        then this number is divided by the set PSM scores mean raised to the number of peptides for that protein
        then we take -Log(X) of the following value.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.down_weighted_multiplicative_log()
        """

        score_list = []
        for proteins in self.pre_score_data:
            cur_scores = proteins.get_psm_scores()
            for scores in cur_scores:
                score_list.append(scores)
        score_mean = numpy.mean(score_list)

        all_scores = []
        logger.info("Scoring Proteins with DWML method")
        for protein in self.pre_score_data:
            val_list = protein.get_psm_scores()
            # Divide by the score mean raised to the length of the number of unique peptides for the protein
            # This is an attempt to normalize for number of peptides per protein
            combine = reduce(lambda x, y: x * y, val_list)
            if combine == 0:
                combine = sys.float_info.min
            score = -math.log(combine / (score_mean ** len(val_list)))
            protein.score = score

            all_scores.append(protein)

        # Higher score is better as a smaller q or pep in a -log will give a larger value
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)

        self.data.protein_score = self.DOWNWEIGHTED_MULTIPLICATIVE_LOG
        self.data.short_protein_score = self.SHORT_DOWNWEIGHTED_MULTIPLICATIVE_LOG
        self.data.scored_proteins = all_scores

    def top_two_combied(self):
        """
        This method uses a Top Two scoring scheme.
        The top two scores for each protein are multiplied together and we take -Log(X) of the multiplied value.
        If a protein only has 1 score/peptide, then we only do -Log(X) of the 1 peptide score.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.top_two_combied()
        """

        all_scores = []
        logger.info("Scoring Proteins with Top Two Method")
        for protein in self.pre_score_data:
            val_list = protein.get_psm_scores()

            try:
                # Try to combine the top two scores
                # Divide by 2 to attempt to normalize the value
                score = -math.log((val_list[0] * val_list[1]) / 2)
            except IndexError:
                # If there is only 1 score/1 peptide then just use the 1 peptide provided
                score = -math.log(val_list[0])

            protein.score = score
            all_scores.append(protein)

        # Higher score is better as a smaller q or pep in a -log will give a larger value
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)

        self.data.protein_score = self.TOP_TWO_COMBINED
        self.data.short_protein_score = self.SHORT_TOP_TWO_COMBINED
        self.data.scored_proteins = all_scores

    def down_weighted_v2(self):
        """
        This method uses a Downweighted Multiplicative Log scoring scheme.
        Each peptide is iteratively downweighted by raising the peptide QValue or PepValue to the
        following power (1/(1+index_number)).
        Where index_number is the peptide number per protein.
        Each score for a protein provides less and less weight iteratively.

        We also take -Log(X) of the final score here.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.down_weighted_v2()
        """

        all_scores = []
        logger.info("Scoring Proteins with down weighted v2 method")
        for protein in self.pre_score_data:
            val_list = protein.get_psm_scores()

            # Here take each score and raise it to the power of (1/(1+index_number)).
            # This downweights each successive score by reducing its weight in a decreasing fashion
            # Basically, each score for a protein will provide less and less weight iteratively
            val_list = [val_list[x] ** (1 / float(1 + x)) for x in range(len(val_list))]
            # val_list = [val_list[x]**(1/float(1+(float(x)/10))) for x in range(len(val_list))]
            score = -math.log(reduce(lambda x, y: x * y, val_list))

            protein.score = score
            all_scores.append(protein)

        # Higher score is better as a smaller q or pep in a -log will give a larger value
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)

        self.data.protein_score = self.DOWNWEIGHTED_VERSION2
        self.data.short_protein_score = self.SHORT_DOWNWEIGHTED_VERSION2
        self.data.scored_proteins = all_scores

    def iterative_down_weighted_log(self):
        """
        This method uses a Downweighted Multiplicative Log scoring scheme.
        Each peptide is iteratively downweighted by multiplying the peptide QValue or PepValue to
        the following (1+index_number).
        Where index_number is the peptide number per protein.
        Each score for a protein provides less and less weight iteratively.

        We also take -Log(X) of the final score here.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.iterative_down_weighted_log()
        """

        all_scores = []
        logger.info("Scoring Proteins with IDWL method")
        for protein in self.pre_score_data:
            val_list = protein.get_psm_scores()

            # Here take each score and multiply it by its index number).
            # This downweights each successive score by reducing its weight in a decreasing fashion
            # Basically, each score for a protein will provide less and less weight iteratively
            val_list = [val_list[x] * (float(1 + x)) for x in range(len(val_list))]
            # val_list = [val_list[x]**(1/float(1+(float(x)/10))) for x in range(len(val_list))]
            combine = reduce(lambda x, y: x * y, val_list)
            if combine == 0:
                combine = sys.float_info.min
            score = -math.log(combine)
            protein.score = score

            protein.score = score
            all_scores.append(protein)

        # Higher score is better as a smaller q or pep in a -log will give a larger value
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)

        self.data.protein_score = self.ITERATIVE_DOWNWEIGHTED_LOG
        self.data.short_protein_score = self.SHORT_ITERATIVE_DOWNWEIGHTED_LOG
        self.data.scored_proteins = all_scores

    def geometric_mean_log(self):
        """
        This method uses a Geometric Mean scoring scheme.

        We also take -Log(X) of the final score here.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.geometric_mean_log()
        """

        all_scores = []
        logger.info("Scoring Proteins. with GML method")
        for protein in self.pre_score_data:
            psm_scores = protein.get_psm_scores()
            val_list = []
            for vals in psm_scores:
                val_list.append(float(vals))
                combine = reduce(lambda x, y: x * y, val_list)
                if combine == 0:
                    combine = sys.float_info.min
                pre_log_score = combine ** (1 / float(len(val_list)))
            score = -math.log(pre_log_score)

            protein.score = score
            all_scores.append(protein)

        # Higher score is better as a smaller q or pep in a -log will give a larger value
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)

        self.data.protein_score = self.GEOMETRIC_MEAN
        self.data.short_protein_score = self.SHORT_GEOMETRIC_MEAN
        self.data.scored_proteins = all_scores

    def iterative_down_weighted_v2(self):
        """
        The following method is an experimental method essentially used for future development of potential scoring
        schemes.
        """

        all_scores = []
        logger.info("Scoring Proteins with iterative down weighted v2 method")
        for protein in self.pre_score_data:
            val_list = protein.get_psm_scores()

            # Here take each score and raise it to the power of (1/(1+index_number)).
            # This downweights each successive score by reducing its weight in a decreasing fashion
            # Basically, each score for a protein will provide less and less weight iteratively
            val_list = [val_list[x] ** (1 / float(1 + x)) for x in range(len(val_list))]
            # val_list = [val_list[x]**(1/float(1+(float(x)/10))) for x in range(len(val_list))]
            score = -math.log(reduce(lambda x, y: x * y, val_list))

            protein.score = score
            all_scores.append(protein)

        # Higher score is better as a smaller q or pep in a -log will give a larger value
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)

        self.data.protein_score = "iterative_downweighting2"
        self.data.short_protein_score = "idw2"
        self.data.scored_proteins = all_scores

    def additive(self):
        """
        This method uses an additive scoring scheme.
        The method can only be used if a larger PSM score is a better PSM score such as the Percolator score.

        Examples:
            >>> score = pyproteininference.scoring.Score(data=data)
            >>> score.additive()
        """

        all_scores = []
        logger.info("Scoring Proteins with additive method")
        for protein in self.pre_score_data:
            val_list = protein.get_psm_scores()

            # Take the sum of our scores
            score = sum(val_list)

            protein.score = score
            all_scores.append(protein)

        # Higher score is better as a smaller q or pep in a -log will give a larger value
        all_scores = sorted(all_scores, key=lambda k: k.score, reverse=True)

        self.data.protein_score = self.ADDITIVE
        self.data.short_protein_score = self.SHORT_ADDITIVE
        self.data.scored_proteins = all_scores
