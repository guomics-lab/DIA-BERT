import logging
import sys
from collections import OrderedDict

import pulp

from src.pyproteininference import datastore
from src.pyproteininference.physical import ProteinGroup, Psm

logger = logging.getLogger(__name__)

# set up our logger
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class Inference(object):
    """
    Parent Inference class for all inference/grouper subset classes.
    The base Inference class contains several methods that are shared across the Inference sub-classes.

    Attributes:
        data (DataStore): [DataStore object][pyproteininference.datastore.DataStore].
        digest (Digest): [Digest object][pyproteininference.in_silico_digest.Digest].
    """

    PARSIMONY = "parsimony"
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"
    FIRST_PROTEIN = "first_protein"
    PEPTIDE_CENTRIC = "peptide_centric"

    INFERENCE_TYPES = [
        PARSIMONY,
        INCLUSION,
        EXCLUSION,
        FIRST_PROTEIN,
        PEPTIDE_CENTRIC,
    ]

    INFERENCE_NAME_MAP = {
        PARSIMONY: "Parsimony",
        INCLUSION: "Inclusion",
        EXCLUSION: "Exclusion",
        FIRST_PROTEIN: "First Protein",
        PEPTIDE_CENTRIC: "Peptide Centric",
    }

    SUBSET_PEPTIDES = "subset_peptides"
    SHARED_PEPTIDES = "shared_peptides"
    NONE_GROUPING = None

    GROUPING_TYPES = [SUBSET_PEPTIDES, SHARED_PEPTIDES, NONE_GROUPING]

    PULP = "pulp"
    LP_SOLVERS = [PULP]

    ALL_SHARED_PEPTIDES = "all"
    BEST_SHARED_PEPTIDES = "best"
    NONE_SHARED_PEPTIDES = None
    SHARED_PEPTIDE_TYPES = [
        ALL_SHARED_PEPTIDES,
        BEST_SHARED_PEPTIDES,
        NONE_SHARED_PEPTIDES,
    ]

    def __init__(self, data, digest):
        """
        Initialization method of Inference object.

        Args:
            data (DataStore): [DataStore object][pyproteininference.datastore.DataStore].
            digest (Digest): [Digest object][pyproteininference.in_silico_digest.Digest].

        """
        self.data = data
        self.digest = digest
        self.data._validate_scored_proteins()

    @classmethod
    def run_inference(cls, data, digest):
        """
        This class method dispatches to one of the five different inference classes/models
        based on input from the [ProteinInferenceParameter][pyproteininference.parameters.ProteinInferenceParameter]
        object.
        The methods are "parsimony", "inclusion", "exclusion", "peptide_centric", and "first_protein".

        Args:
            data (DataStore): [DataStore object][pyproteininference.datastore.DataStore].
            digest (Digest): [Digest object][pyproteininference.in_silico_digest.Digest].

        Example:
            >>> pyproteininference.inference.Inference.run_inference(data=data,digest=digest)

        """

        inference_type = data.parameter_file_object.inference_type

        logger.info("Running Inference with Inference Type: {}".format(inference_type))

        if inference_type == Inference.PARSIMONY:
            group = Parsimony(data=data, digest=digest)
            group.infer_proteins()

        if inference_type == Inference.INCLUSION:
            group = Inclusion(data=data, digest=digest)
            group.infer_proteins()

        if inference_type == Inference.EXCLUSION:
            group = Exclusion(data=data, digest=digest)
            group.infer_proteins()

        if inference_type == Inference.FIRST_PROTEIN:
            group = FirstProtein(data=data, digest=digest)
            group.infer_proteins()

        if inference_type == Inference.PEPTIDE_CENTRIC:
            group = PeptideCentric(data=data, digest=digest)
            group.infer_proteins()

    def _create_protein_groups(self, scored_proteins):
        """
        This method sets up protein groups for inference methods that do not need grouping.

        Args:
            scored_proteins (list): List of scored [Protein][pyproteininference.physical.Protein] objects.

        Returns:
            list: List of lists of scored [Protein][pyproteininference.physical.Protein] objects.

        """
        scored_proteins = sorted(
            scored_proteins,
            key=lambda k: (k.score, len(k.raw_peptides), k.identifier),
            reverse=True,
        )

        prot_pep_dict = self.data.protein_to_peptide_dictionary()

        restricted_peptides_set = set(self.data.restricted_peptides)

        grouped_proteins = []
        for protein_objects in scored_proteins:
            cur_protein_identifier = protein_objects.identifier

            # Set peptide variable if the peptide is in the restricted peptide set
            # Sort the peptides alphabetically
            protein_objects.peptides = set(
                sorted([x for x in prot_pep_dict[cur_protein_identifier] if x in restricted_peptides_set])
            )
            protein_list_group = [protein_objects]
            grouped_proteins.append(protein_list_group)
        return grouped_proteins

    def _apply_protein_group_ids(self, grouped_protein_objects):
        """
        This method creates the ProteinGroup objects from the output of
            [_create_protein_groups][`pyproteininference.inference.Inference._create_protein_groups].

        Args:
            grouped_protein_objects (list): list of grouped [Protein][pyproteininference.physical.Protein] objects.

        Returns:
            dict: a Dictionary that contains a list of [ProteinGroup][pyproteininference.physical.ProteinGroup]
                objects (key:"group_objects") and a list of grouped [Protein][pyproteininference.physical.Protein]
                objects (key:"grouped_protein_objects").


        """

        sp_protein_set = set(self.digest.swiss_prot_protein_set)

        prot_pep_dict = self.data.protein_to_peptide_dictionary()

        # Here we create group ID's
        group_id = 0
        protein_group_objects = []
        for protein_group in grouped_protein_objects:
            protein_list = []
            group_id = group_id + 1
            pg = ProteinGroup(group_id)
            logger.debug("Created Protein Group with ID: {}".format(str(group_id)))
            for protein in protein_group:
                cur_protein = protein
                # The following loop assigns group_id's, reviewed/unreviewed status, and number of unique peptides...
                if group_id not in cur_protein.group_identification:
                    cur_protein.group_identification.add(group_id)
                if protein.identifier in sp_protein_set:
                    cur_protein.reviewed = True
                else:
                    cur_protein.unreviewed = True
                cur_identifier = protein.identifier
                cur_protein.num_peptides = len(prot_pep_dict[cur_identifier])
                # Here append the number of unique peptides... so we can use this as secondary sorting...
                protein_list.append(cur_protein)
                # Sorted protein_groups then becomes a list of lists... of protein objects

            pg.proteins = protein_list
            protein_group_objects.append(pg)

        return_dict = {
            "grouped_protein_objects": grouped_protein_objects,
            "group_objects": protein_group_objects,
        }

        return return_dict


class Inclusion(Inference):
    """
    Inclusion Inference class. This class contains methods that support the initialization of an
    Inclusion inference method.

    Attributes:
        data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
        digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].
        scored_data (list): a List of scored [Protein][pyproteininference.physical.Protein] objects.

    """

    def __init__(self, data, digest):
        """
        Initialization method of the Inclusion Inference method.

        Args:
            data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
            digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].
        """

        self.data = data
        self.digest = digest
        self.data._validate_scored_proteins()
        self.scored_data = self.data.get_protein_data()

    def infer_proteins(self):
        """
        This method performs the grouping for Inclusion.

        Inclusion actually does not do grouping as all peptides get assigned to all possible proteins
        and groups are not created.

        This method assigns the variables: `grouped_scored_proteins` and `protein_group_objects`.
        These are both variables of the [DataStore Object][pyproteininference.datastore.DataStore] and are
        lists of [Protein][pyproteininference.physical.Protein] objects
        and [ProteinGroup][pyproteininference.physical.ProteinGroup] objects.
        """

        grouped_proteins = self._create_protein_groups(scored_proteins=self.scored_data)

        hl = self.data.higher_or_lower()

        logger.info("Applying Group ID's for the Inclusion Method")

        regrouped_proteins = self._apply_protein_group_ids(
            grouped_protein_objects=grouped_proteins,
        )

        grouped_protein_objects = regrouped_proteins["grouped_protein_objects"]
        protein_group_objects = regrouped_proteins["group_objects"]

        logger.info("Sorting Results based on lead Protein Score")
        grouped_protein_objects = datastore.DataStore.sort_protein_objects(
            grouped_protein_objects=grouped_protein_objects, higher_or_lower=hl
        )
        protein_group_objects = datastore.DataStore.sort_protein_group_objects(
            protein_group_objects=protein_group_objects, higher_or_lower=hl
        )

        self.data.grouped_scored_proteins = grouped_protein_objects
        self.data.protein_group_objects = protein_group_objects

    def _apply_protein_group_ids(self, grouped_protein_objects):
        """
        This method creates the ProteinGroup objects for the inclusion inference type using protein groups from
         [_create_protein_groups][`pyproteininference.inference.Inference._create_protein_groups].

        Args:
            grouped_protein_objects (list): list of grouped [Protein][pyproteininference.physical.Protein] objects.

        Returns:
            dict: a Dictionary that contains a list of [ProteinGroup][pyproteininference.physical.ProteinGroup]
                objects (key:"group_objects") and a list of
                grouped [Protein][pyproteininference.physical.Protein] objects (key:"grouped_protein_objects").

        """

        sp_protein_set = set(self.digest.swiss_prot_protein_set)

        prot_pep_dict = self.data.protein_to_peptide_dictionary()

        # Here we create group ID's
        group_id = 0
        protein_group_objects = []
        for protein_group in grouped_protein_objects:
            protein_list = []
            group_id = group_id + 1
            pg = ProteinGroup(group_id)
            logger.debug("Created Protein Group with ID: {}".format(str(group_id)))
            for prot in protein_group:
                cur_protein = prot
                # The following loop assigns group_id's, reviewed/unreviewed status, and number of unique peptides...
                if group_id not in cur_protein.group_identification:
                    cur_protein.group_identification.add(group_id)
                if cur_protein.identifier in sp_protein_set:
                    cur_protein.reviewed = True
                else:
                    cur_protein.unreviewed = True
                cur_identifier = cur_protein.identifier
                cur_protein.num_peptides = len(prot_pep_dict[cur_identifier])
                # Here append the number of unique peptides... so we can use this as secondary sorting...
                protein_list.append(cur_protein)
                # Sorted protein_groups then becomes a list of lists... of protein objects

            pg.proteins = protein_list
            protein_group_objects.append(pg)

        return_dict = {
            "grouped_protein_objects": grouped_protein_objects,
            "group_objects": protein_group_objects,
        }

        return return_dict


class Exclusion(Inference):
    """
    Exclusion Inference class. This class contains methods that support the initialization of an
    Exclusion inference method.

    Attributes:
        data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
        digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].
        scored_data (list): a List of scored [Protein][pyproteininference.physical.Protein] objects.

    """

    def __init__(self, data, digest):
        """
        Initialization method of the Exclusion Class.

        Args:
            data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
            digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].

        """
        self.data = data
        self.digest = digest
        self.data._validate_scored_proteins()
        self.scored_data = self.data.get_protein_data()
        self.list_of_prots_not_in_db = None
        self.list_of_peps_not_in_db = None

    def infer_proteins(self):
        """
        This method performs the Exclusion inference/grouping method.

        For the exclusion inference method groups cannot be created because all shared peptides are removed.

        This method assigns the variables: `grouped_scored_proteins` and `protein_group_objects`.
        These are both variables of the [DataStore Object][pyproteininference.datastore.DataStore] and are
        lists of [Protein][pyproteininference.physical.Protein] objects
        and [ProteinGroup][pyproteininference.physical.ProteinGroup] objects.

        """

        grouped_proteins = self._create_protein_groups(scored_proteins=self.scored_data)

        hl = self.data.higher_or_lower()

        logger.info("Applying Group ID's for the Exclusion Method")
        regrouped_proteins = self._apply_protein_group_ids(
            grouped_protein_objects=grouped_proteins,
        )

        grouped_protein_objects = regrouped_proteins["grouped_protein_objects"]
        protein_group_objects = regrouped_proteins["group_objects"]

        logger.info("Sorting Results based on lead Protein Score")
        grouped_protein_objects = datastore.DataStore.sort_protein_objects(
            grouped_protein_objects=grouped_protein_objects, higher_or_lower=hl
        )
        protein_group_objects = datastore.DataStore.sort_protein_group_objects(
            protein_group_objects=protein_group_objects, higher_or_lower=hl
        )

        self.data.grouped_scored_proteins = grouped_protein_objects
        self.data.protein_group_objects = protein_group_objects


class Parsimony(Inference):
    """
    Parsimony Inference class. This class contains methods that support the initialization of a
    Parsimony inference method.

    Attributes:
        data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
        digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].
        scored_data (list): a List of scored [Protein][pyproteininference.physical.Protein] objects.
        lead_protein_set (set): Set of protein strings that are classified as leads from the LP solver.

    """

    def __init__(self, data, digest):
        """
        Initialization method of the Parsimony object.

        Args:
            data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
            digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].
        """
        self.data = data
        self.digest = digest
        self.data._validate_scored_proteins()
        self.scored_data = self.data.get_protein_data()
        self.lead_protein_set = None
        self.parameter_file_object = data.parameter_file_object

    def _create_protein_groups(
        self,
        all_scored_proteins,
        lead_protein_objects,
        grouping_type="shared_peptides",
    ):
        """
        Internal method that creates a list of lists of [Protein][pyproteininference.physical.Protein]
        objects for the Parsimony inference object.
        These list of lists are "groups" and the proteins get grouped them according to grouping_type variable.

        Args:
            all_scored_proteins (list): list of [Protein][pyproteininference.physical.Protein] objects.
            lead_protein_objects (list): list of [Protein][pyproteininference.physical.Protein] objects
                Only needed if inference_type=parsimony.
            grouping_type: (str): One of `GROUPING_TYPES`.

        Returns:
            list: list of lists of [Protein][pyproteininference.physical.Protein] objects.

        """

        logger.info("Grouping Peptides with Grouping Type: {}".format(grouping_type))
        logger.info("Grouping Peptides with Inference Type: {}".format(self.PARSIMONY))

        all_scored_proteins = sorted(
            all_scored_proteins,
            key=lambda k: (len(k.raw_peptides), k.identifier),
            reverse=True,
        )

        lead_scored_proteins = lead_protein_objects
        lead_scored_proteins = sorted(
            lead_scored_proteins,
            key=lambda k: (len(k.raw_peptides), k.identifier),
            reverse=True,
        )

        protein_finder = [x.identifier for x in all_scored_proteins]

        prot_pep_dict = self.data.protein_to_peptide_dictionary()

        protein_tracker = set()
        restricted_peptides_set = set(self.data.restricted_peptides)
        try:
            picked_removed = set([x.identifier for x in self.data.picked_proteins_removed])
        except TypeError:
            picked_removed = set()

        missing_proteins = set()
        in_silico_peptides_to_proteins = self.digest.peptide_to_protein_dictionary
        grouped_proteins = []
        for protein_objects in lead_scored_proteins:
            if protein_objects not in protein_tracker:
                protein_tracker.add(protein_objects)
                cur_protein_identifier = protein_objects.identifier

                # Set peptide variable if the peptide is in the restricted peptide set
                # Sort the peptides alphabetically
                protein_objects.peptides = set(
                    sorted([x for x in prot_pep_dict[cur_protein_identifier] if x in restricted_peptides_set])
                )
                protein_list_group = [protein_objects]
                current_peptides = prot_pep_dict[cur_protein_identifier]

                current_grouped_proteins = set()
                for (
                    peptide
                ) in current_peptides:  # Probably put an if here... if peptide is in the list of peptide after being
                    # restricted by datastore.RestrictMainData
                    if peptide in restricted_peptides_set:
                        # Get the proteins that map to the current peptide using in_silico_peptides_to_proteins
                        # First make sure our peptide is formatted properly...
                        if not peptide.isupper() or not peptide.isalpha():
                            # If the peptide is not all upper case or if its not all alphabetical...
                            peptide = Psm.remove_peptide_mods(peptide)
                        potential_protein_list = in_silico_peptides_to_proteins[peptide]
                        if not potential_protein_list:
                            logger.warning(
                                "Protein {} and Peptide {} is not in database...".format(
                                    protein_objects.identifier, peptide
                                )
                            )

                        # Assign proteins to groups based on shared peptide... unless the protein is equivalent
                        # to the current identifier
                        if grouping_type != self.NONE_GROUPING:
                            for protein in potential_protein_list:
                                # If statement below to avoid grouping the same protein twice and to not group the lead
                                if (
                                    protein not in current_grouped_proteins
                                    and protein != cur_protein_identifier
                                    and protein not in picked_removed
                                    and protein not in missing_proteins
                                ):
                                    try:
                                        # Try to find its object using protein_finder (list of identifiers) and
                                        # lead_scored_proteins (list of Protein Objects)
                                        cur_index = protein_finder.index(protein)
                                        current_protein_object = all_scored_proteins[cur_index]
                                        if not current_protein_object.peptides:
                                            current_protein_object.peptides = set(
                                                sorted(
                                                    [
                                                        x
                                                        for x in prot_pep_dict[current_protein_object.identifier]
                                                        if x in restricted_peptides_set
                                                    ]
                                                )
                                            )
                                        if grouping_type == self.SHARED_PEPTIDES:
                                            current_grouped_proteins.add(current_protein_object)
                                        elif grouping_type == self.SUBSET_PEPTIDES:
                                            if current_protein_object.peptides.issubset(protein_objects.peptides):
                                                current_grouped_proteins.add(current_protein_object)
                                                protein_tracker.add(current_protein_object)
                                            else:
                                                pass
                                        else:
                                            pass
                                    except ValueError:
                                        logger.warning(
                                            "Protein from DB {} not found with protein finder for peptide {}".format(
                                                protein, peptide
                                            )
                                        )
                                        missing_proteins.add(protein)

                                else:
                                    pass
                # Add the proteins to the lead if they share peptides...
                protein_list_group = protein_list_group + list(current_grouped_proteins)
                # protein_list_group at first is just the lead protein object...
                # We then try apply grouping by looking at all peptide from the lead...
                # For all of these peptide look at all other non lead proteins and try to assign them to the group...
                # We assign the entire protein object as well... in the above try/except
                # Then append this sub group to the main list
                # The variable grouped_proteins is now a list of lists which each element being a Protein object and
                # each list of protein objects corresponding to a group
                grouped_proteins.append(protein_list_group)

        return grouped_proteins

    def _swissprot_and_isoform_override(
        self,
        scored_data,
        grouped_proteins,
        override_type="soft",
        isoform_override=True,
    ):
        """
        This internal method creates and reorders protein groups based on criteria such as Reviewed/Unreviewed
        Identifiers as well as Canonincal/Isoform Identifiers.
        This method is only used with parsimony inference type.

        Args:
            scored_data (list): list of scored [Protein][pyproteininference.physical.Protein] objects.
            grouped_proteins:  list of grouped [Protein][pyproteininference.physical.Protein] objects.
            override_type (str): "soft" or "hard" to indicate Reviewed/Unreviewed override. "soft" is preferred and
                default.
            isoform_override (bool): True/False on whether to favor canonical forms vs isoforms as group leads.

        Returns:
            dict: a Dictionary that contains a list of [ProteinGroup][pyproteininference.physical.ProteinGroup] objects
            (key:"group_objects") and a list of grouped [Protein][pyproteininference.physical.Protein]
            objects (key:"grouped_protein_objects").


        """

        sp_protein_set = set(self.digest.swiss_prot_protein_set)
        scored_proteins = list(scored_data)
        protein_finder = [x.identifier for x in scored_proteins]

        prot_pep_dict = self.data.protein_to_peptide_dictionary()

        # Get the higher or lower variable
        higher_or_lower = self.data.higher_or_lower()

        logger.info("Applying Group IDs... and Executing {} Swissprot Override...".format(override_type))
        # Here we create group ID's for all groups and do some sorting
        grouped_protein_objects = []
        group_id = 0
        leads = set()
        protein_group_objects = []
        for protein_group in grouped_proteins:
            protein_list = []
            group_id = group_id + 1
            # Make a protein group
            pg = ProteinGroup(group_id)
            logger.debug("Created Protein Group with ID: {}".format(str(group_id)))
            for prots in protein_group:
                # Loop over all proteins in the original group
                try:
                    # The following loop assigns group_id's, reviewed/unreviewed status, and number of unique peptides
                    pindex = protein_finder.index(prots.identifier)
                    # Attempt to find the protein object by identifier
                    cur_protein = scored_proteins[pindex]
                    if group_id not in cur_protein.group_identification:
                        cur_protein.group_identification.add(group_id)
                    if prots.identifier in sp_protein_set:
                        cur_protein.reviewed = True
                    else:
                        cur_protein.unreviewed = True
                    cur_identifier = prots.identifier
                    cur_protein.num_peptides = len(prot_pep_dict[cur_identifier])
                    # Here append the number of unique peptides... so we can use this as secondary sorting...
                    protein_list.append(cur_protein)
                    # Sorted groups then becomes a list of lists... of protein objects

                except ValueError:
                    # Here we pass if the protein does not have a score...
                    # Potentially it got 'picked' (removed) by protein picker...
                    pass

            # Sort protein sub group
            protein_list = datastore.DataStore.sort_protein_sub_groups(
                protein_list=protein_list, higher_or_lower=higher_or_lower
            )

            # grouped_protein_objects is the MAIN list of lists with grouped protein objects
            grouped_protein_objects.append(protein_list)
            # If the lead is reviewed append it to leads and do nothing else...
            # If the lead is unreviewed then try to replace it with the best reviewed hit
            # Run swissprot override
            if self.data.parameter_file_object.reviewed_identifier_symbol:
                sp_override = self._swissprot_override(
                    protein_list=protein_list,
                    leads=leads,
                    grouped_protein_objects=grouped_protein_objects,
                    override_type=override_type,
                )
                grouped_protein_objects = sp_override["grouped_protein_objects"]
                leads = sp_override["leads"]
                protein_list = sp_override["protein_list"]

            # Run isoform override If we want to run isoform_override and if the isoform symbol exists...
            if isoform_override and self.data.parameter_file_object.isoform_symbol:
                iso_override = self._isoform_override(
                    protein_list=protein_list,
                    leads=leads,
                    grouped_protein_objects=grouped_protein_objects,
                )
                grouped_protein_objects = iso_override["grouped_protein_objects"]
                leads = iso_override["leads"]
                protein_list = iso_override["protein_list"]

            pg.proteins = protein_list
            protein_group_objects.append(pg)

        return_dict = {
            "grouped_protein_objects": grouped_protein_objects,
            "group_objects": protein_group_objects,
        }

        return return_dict

    def _swissprot_override(self, protein_list, leads, grouped_protein_objects, override_type):
        """
        This method re-assigns protein group leads if the lead is an unreviewed protein and if the protein group
         contains a reviewed protein that contains the exact same set of peptides as the unreviewed lead.
        This method is here to provide consistency to the output.

        Args:
            protein_list (list): List of grouped [Protein][pyproteininference.physical.Protein] objects.
            leads (set): Set of string protein identifiers that have been identified as a lead.
            grouped_protein_objects (list): List of protein_list lists.
            override_type (str): "soft" or "hard" on how to override non reviewed identifiers. "soft" is preferred.

        Returns:
            dict: leads (set): Set of string protein identifiers that have been identified as a lead.
             Updated to reflect lead changes.
            grouped_protein_objects (list): List of protein_list lists. Updated to reflect lead changes.
            protein_list (list): List of grouped [Protein][pyproteininference.physical.Protein] objects.
                Updated to reflect lead changes.

        """

        if not protein_list[0].reviewed:
            # If the lead is unreviewed attempt to replace it...
            # Start to loop through protein_list which is the current group...
            for protein in protein_list[1:]:
                # Find the first reviewed it... if its not a lead protein already then do score swap and break...
                if protein.reviewed:
                    best_swiss_prot_prot = protein

                    if override_type == "soft":
                        # If the lead proteins peptides are a subset of the best swissprot.... then swap the proteins.
                        # (meaning equal peptides or the swissprot completely covers the tremble reference)
                        if best_swiss_prot_prot.identifier not in leads and set(protein_list[0].peptides).issubset(
                            set(best_swiss_prot_prot.peptides)
                        ):
                            # We use -1 as the idex of grouped_protein_objects because the current 'protein_list' is
                            # the last entry appended to scores grouped
                            # Essentially grouped_protein_objects[-1]==protein_list
                            # We need this syntax so we can switch the location of the unreviewed lead identifier with
                            # the best reviewed identifier in grouped_protein_objects
                            swiss_prot_override_index = grouped_protein_objects[-1].index(best_swiss_prot_prot)
                            cur_tr_lead = grouped_protein_objects[-1][0]
                            (
                                grouped_protein_objects[-1][0],
                                grouped_protein_objects[-1][swiss_prot_override_index],
                            ) = (
                                grouped_protein_objects[-1][swiss_prot_override_index],
                                grouped_protein_objects[-1][0],
                            )
                            grouped_protein_objects[-1][swiss_prot_override_index], grouped_protein_objects[-1][0]
                            new_sp_lead = grouped_protein_objects[-1][0]
                            logger.info(
                                "Overriding Unreviewed {} with Reviewed {}".format(
                                    cur_tr_lead.identifier, new_sp_lead.identifier
                                )
                            )

                            # Append new_sp_lead protein to leads, to make sure we dont repeat leads
                            leads.add(new_sp_lead.identifier)
                            break
                        else:
                            # If no reviewed and none not in leads then pass...
                            pass

                    if override_type == "hard":
                        if best_swiss_prot_prot.identifier not in leads:
                            # We use -1 as the index of grouped_protein_objects because the current 'protein_list'
                            # is the last entry appended to grouped_protein_objects
                            # Essentially grouped_protein_objects[-1]==protein_list
                            # We need this syntax so we can switch the location of the unreviewed lead identifier
                            # with the best reviewed identifier in grouped_protein_objects
                            swiss_prot_override_index = grouped_protein_objects[-1].index(best_swiss_prot_prot)
                            cur_tr_lead = grouped_protein_objects[-1][0]
                            # Re-assigning the value within the index will also reassign the value in protein_list...
                            # This is because grouped_protein_objects[-1] equals protein_list
                            # So we do not have to reassign values in protein_list
                            (
                                grouped_protein_objects[-1][0],
                                grouped_protein_objects[-1][swiss_prot_override_index],
                            ) = (
                                grouped_protein_objects[-1][swiss_prot_override_index],
                                grouped_protein_objects[-1][0],
                            )
                            new_sp_lead = grouped_protein_objects[-1][0]
                            logger.info(
                                "Overriding Unreviewed {} with Reviewed {}".format(
                                    cur_tr_lead.identifier, new_sp_lead.identifier
                                )
                            )

                            # Append new_sp_lead protein to leads, to make sure we dont repeat leads
                            leads.add(new_sp_lead.identifier)
                            break
                        else:
                            # If no reviewed and none not in leads then pass...
                            pass

                else:
                    pass

        else:
            leads.add(protein_list[0].identifier)

        return_dict = {
            "leads": leads,
            "grouped_protein_objects": grouped_protein_objects,
            "protein_list": protein_list,
        }

        return return_dict

    def _isoform_override(self, protein_list, grouped_protein_objects, leads):
        """
        This method re-assigns protein group leads if the lead is an isoform protein and if the protein group contains
        a canonical protein that contains the exact same set of peptides as the isoform lead.
        This method is here to provide consistency to the output.

        Args:
            protein_list (list): List of grouped [Protein][pyproteininference.physical.Protein] objects.
            leads (set): Set of string protein identifiers that have been identified as a lead.
            grouped_protein_objects (list): List of protein_list lists.

        Returns:
            dict: leads (set): Set of string protein identifiers that have been identified as a lead. Updated to
                reflect lead changes.
            grouped_protein_objects (list): List of protein_list lists. Updated to reflect lead changes.
            protein_list (list): List of grouped [Protein][pyproteininference.physical.Protein] objects.
                Updated to reflect lead changes.


        """

        if self.data.parameter_file_object.isoform_symbol in protein_list[0].identifier:
            pure_id = protein_list[0].identifier.split(self.data.parameter_file_object.isoform_symbol)[0]
            # Start to loop through protein_list which is the current group...
            for potential_replacement in protein_list[1:]:
                isoform_override = potential_replacement
                if (
                    isoform_override.identifier == pure_id
                    and isoform_override.identifier not in leads
                    and set(protein_list[0].peptides).issubset(set(isoform_override.peptides))
                ):
                    isoform_override_index = grouped_protein_objects[-1].index(isoform_override)
                    cur_iso_lead = grouped_protein_objects[-1][0]
                    # Re-assigning the value within the index will also reassign the value in protein_list...
                    # This is because grouped_protein_objects[-1] equals protein_list
                    # So we do not have to reassign values in protein_list
                    (grouped_protein_objects[-1][0], grouped_protein_objects[-1][isoform_override_index],) = (
                        grouped_protein_objects[-1][isoform_override_index],
                        grouped_protein_objects[-1][0],
                    )
                    grouped_protein_objects[-1][isoform_override_index], grouped_protein_objects[-1][0]

                    new_iso_lead = grouped_protein_objects[-1][0]
                    logger.info(
                        "Overriding Isoform {} with {}".format(cur_iso_lead.identifier, new_iso_lead.identifier)
                    )
                    leads.add(protein_list[0].identifier)

        return_dict = {
            "leads": leads,
            "grouped_protein_objects": grouped_protein_objects,
            "protein_list": protein_list,
        }

        return return_dict

    def _reassign_protein_group_leads(self, protein_group_objects):
        """
        This internal method corrects leads that are improperly assigned in the parsimony inference method.
        This method acts on the protein group objects.

        Args:
            protein_group_objects (list): List of [ProteinGroup][pyproteininference.physical.ProteinGroup] objects.

        Returns:
            protein_group_objects (list): List of [ProteinGroup][pyproteininference.physical.ProteinGroup] objects
            where leads have been reassigned properly.


        """

        # Get the higher or lower variable
        if not self.data.high_low_better:
            higher_or_lower = self.data.higher_or_lower()
        else:
            higher_or_lower = self.data.high_low_better

        # Sometimes we have cases where:
        # protein a maps to peptides 1,2,3
        # protein b maps to peptides 1,2
        # protein c maps to a bunch of peptides and peptide 3
        # Therefore, in the model proteins a and b are equivalent in that they map to 2 peptides together - 1 and 2.
        # peptide 3 maps to a but also to c...
        # Sometimes the model (pulp) will spit out protein b as the lead... we wish to swap protein b as the lead with
        # protein a because it will likely have a better score...
        logger.info("Potentially Reassigning Protein Group leads...")
        lead_protein_set = set([x.proteins[0].identifier for x in protein_group_objects])
        for i in range(len(protein_group_objects)):
            for j in range(1, len(protein_group_objects[i].proteins)):  # Loop over all sub proteins in the group...
                # if the lead proteins peptides are a subset of one of its proteins in the group, and the secondary
                # protein is not a lead protein and its score is better than the leads... and it has more peptides...
                new_lead = protein_group_objects[i].proteins[j]
                old_lead = protein_group_objects[i].proteins[0]
                if higher_or_lower == datastore.DataStore.HIGHER_PSM_SCORE:
                    if (
                        set(old_lead.peptides).issubset(set(new_lead.peptides))
                        and new_lead.identifier not in lead_protein_set
                        and old_lead.score <= new_lead.score
                        and len(old_lead.peptides) < len(new_lead.peptides)
                    ):
                        logger.info(
                            "protein {} will replace protein {} as lead, with index {}, New Num Peptides: {}, "
                            "Old Num Peptides: {}".format(
                                str(new_lead.identifier),
                                str(old_lead.identifier),
                                str(j),
                                str(len(new_lead.peptides)),
                                str(len(old_lead.peptides)),
                            )
                        )
                        lead_protein_set.add(new_lead.identifier)
                        lead_protein_set.remove(old_lead.identifier)
                        # Swap their positions in the list
                        (
                            protein_group_objects[i].proteins[0],
                            protein_group_objects[i].proteins[j],
                        ) = (new_lead, old_lead)
                        break

                if higher_or_lower == datastore.DataStore.LOWER_PSM_SCORE:
                    if (
                        set(old_lead.peptides).issubset(set(new_lead.peptides))
                        and new_lead.identifier not in lead_protein_set
                        and old_lead.score >= new_lead.score
                        and len(old_lead.peptides) < len(new_lead.peptides)
                    ):
                        logger.info(
                            "protein {} will replace protein {} as lead, with index {}, New Num Peptides: {}, "
                            "Old Num Peptides: {}".format(
                                str(new_lead.identifier),
                                str(old_lead.identifier),
                                str(j),
                                str(len(new_lead.peptides)),
                                str(len(old_lead.peptides)),
                            )
                        )
                        lead_protein_set.add(new_lead.identifier)
                        lead_protein_set.remove(old_lead.identifier)
                        # Swap their positions in the list
                        (
                            protein_group_objects[i].proteins[0],
                            protein_group_objects[i].proteins[j],
                        ) = (new_lead, old_lead)
                        break

        return protein_group_objects

    def _reassign_protein_list_leads(self, grouped_protein_objects):
        """
        This internal method corrects leads that are improperly assigned in the parsimony inference method.
        This method acts on the grouped protein objects.

        Args:
            grouped_protein_objects (list): List of [Protein][pyproteininference.physical.Protein] objects.

        Returns:
            list: List of [Protein][pyproteininference.physical.Protein] objects where leads have been
                reassigned properly.


        """

        # Get the higher or lower variable
        if not self.data.high_low_better:
            higher_or_lower = self.data.higher_or_lower()
        else:
            higher_or_lower = self.data.high_low_better

        # Sometimes we have cases where:
        # protein a maps to peptides 1,2,3
        # protein b maps to peptides 1,2
        # protein c maps to a bunch of peptides and peptide 3
        # Therefore, in the model proteins a and b are equivalent in that they map to 2 peptides together - 1 and 2.
        # peptide 3 maps to a but also to c...
        # Sometimes the model (pulp) will spit out protein b as the lead... we wish to swap protein b as the lead with
        # protein a because it will likely have a better score...
        logger.info("Potentially Reassigning Proteoin List leads...")
        lead_protein_set = set([x[0].identifier for x in grouped_protein_objects])
        for i in range(len(grouped_protein_objects)):
            for j in range(1, len(grouped_protein_objects[i])):  # Loop over all sub proteins in the group...
                # if the lead proteins peptides are a subset of one of its proteins in the group, and the secondary
                # protein is not a lead protein and its score is better than the leads... and it has more peptides...
                new_lead = grouped_protein_objects[i][j]
                old_lead = grouped_protein_objects[i][0]
                if higher_or_lower == datastore.DataStore.HIGHER_PSM_SCORE:
                    if (
                        set(old_lead.peptides).issubset(set(new_lead.peptides))
                        and new_lead.identifier not in lead_protein_set
                        and old_lead.score <= new_lead.score
                        and len(old_lead.peptides) < len(new_lead.peptides)
                    ):
                        logger.info(
                            "protein {} will replace protein {} as lead, with index {}, New Num Peptides: {}, "
                            "Old Num Peptides: {}".format(
                                str(new_lead.identifier),
                                str(old_lead.identifier),
                                str(j),
                                str(len(new_lead.peptides)),
                                str(len(old_lead.peptides)),
                            )
                        )
                        lead_protein_set.add(new_lead.identifier)
                        lead_protein_set.remove(old_lead.identifier)
                        # Swap their positions in the list
                        (
                            grouped_protein_objects[i][0],
                            grouped_protein_objects[i][j],
                        ) = (new_lead, old_lead)
                        break

                if higher_or_lower == datastore.DataStore.LOWER_PSM_SCORE:
                    if (
                        set(old_lead.peptides).issubset(set(new_lead.peptides))
                        and new_lead.identifier not in lead_protein_set
                        and old_lead.score >= new_lead.score
                        and len(old_lead.peptides) < len(new_lead.peptides)
                    ):
                        logger.info(
                            "protein {} will replace protein {} as lead, with index {}, New Num Peptides: {}, "
                            "Old Num Peptides: {}".format(
                                str(new_lead.identifier),
                                str(old_lead.identifier),
                                str(j),
                                str(len(new_lead.peptides)),
                                str(len(old_lead.peptides)),
                            )
                        )
                        lead_protein_set.add(new_lead.identifier)
                        lead_protein_set.remove(old_lead.identifier)
                        # Swap their positions in the list
                        (
                            grouped_protein_objects[i][0],
                            grouped_protein_objects[i][j],
                        ) = (new_lead, old_lead)
                        break

        return grouped_protein_objects

    def _pulp_grouper(self):
        """
        This internal function uses pulp to solve the lp problem for parsimony then performs protein grouping with the
         various internal grouping functions.

        This method assigns the variables: `grouped_scored_proteins` and `protein_group_objects`.
        These are both variables of the [DataStore object][pyproteininference.datastore.DataStore] and are
        lists of [Protein][pyproteininference.physical.Protein] objects
        and [ProteinGroup][pyproteininference.physical.ProteinGroup] objects.

        """

        # Here we get the peptide to protein dictionary
        pep_prot_dict = self.data.peptide_to_protein_dictionary()

        self.data.protein_to_peptide_dictionary()

        identifiers_sorted = self.data.get_sorted_identifiers(scored=True)

        # Get all the proteins that we scored and the ones picked if picker was ran...
        data_proteins = sorted([x for x in self.data.protein_peptide_dictionary.keys() if x in identifiers_sorted])
        # Get the set of peptides for each protein...
        data_peptides = [set(self.data.protein_peptide_dictionary[x]) for x in data_proteins]
        flat_peptides_in_data = set([item for sublist in data_peptides for item in sublist])

        peptide_sets = []
        # Loop over the list of peptides...
        for k in range(len(data_peptides)):
            raw_peptides = data_peptides[k]
            peptide_set = set()
            # Loop over each individual peptide per protein...
            for peps in raw_peptides:
                peptide = peps

                # Remove mods...
                new_peptide = Psm.remove_peptide_mods(peptide)
                # Add it to a temporary set...
                peptide_set.add(new_peptide)
            # Append this set to a new list...
            peptide_sets.append(peptide_set)
            # Set that proteins peptides to be the unmodified ones...
            data_peptides[k] = peptide_set

        # Get them all...
        all_peptides = [x for x in data_peptides]
        # Remove redundant sets...
        non_redundant_peptide_sets = [set(i) for i in OrderedDict.fromkeys(frozenset(item) for item in peptide_sets)]

        # Loop over  the restricted list of peptides...
        ind_list = []
        for pep_sets in non_redundant_peptide_sets:
            # Get its index in terms of the overall list...
            ind_list.append(all_peptides.index(pep_sets))

        # Get the protein based on the index
        restricted_proteins = [data_proteins[x] for x in range(len(data_peptides)) if x in ind_list]

        # Here we get the list of all proteins
        plist = []
        for peps in pep_prot_dict.keys():
            for prots in list(pep_prot_dict[peps]):
                if prots in restricted_proteins and peps in flat_peptides_in_data:
                    plist.append(prots)

        # Here we get the unique proteins
        unique_prots = list(set(plist).union())
        unique_protein_set = set(unique_prots)

        unique_prots_sorted = [x for x in identifiers_sorted if x in unique_prots]

        # Define the protein variables with a lower bound of 0 and catgeory Integer
        prots = pulp.LpVariable.dicts("prot", indices=unique_prots_sorted, lowBound=0, cat="Integer")

        # Define our Lp Problem which is to Minimize our objective function
        prob = pulp.LpProblem("Parsimony_Problem", pulp.LpMinimize)

        # Define our objective function, which is to take the sum of all of our proteins and find the minimum set.
        prob += pulp.lpSum([prots[i] for i in prots])

        # Set up our constraints. The constrains are as follows:

        # Loop over each peptide and determine the proteins it maps to...
        # Each peptide is a constraint with the proteins it maps to having to be greater than or equal to 1
        # In the case below we see that protein 3 has a unique peptide, protein 2 is redundant

        logger.info("Sorting peptides before looping")
        for peptides in sorted(list(pep_prot_dict.keys())):
            try:
                prob += (
                    pulp.lpSum([prots[i] for i in sorted(list(pep_prot_dict[peptides])) if i in unique_protein_set])
                    >= 1
                )
            except KeyError:
                logger.info("Not including protein {} in pulp model".format(pep_prot_dict[peptides]))

        prob.solve()

        scored_data = self.data.get_protein_data()
        scored_proteins = list(scored_data)
        protein_finder = [x.identifier for x in scored_proteins]

        lead_protein_objects = []
        lead_protein_identifiers = []
        for proteins in unique_prots_sorted:
            parsimony_value = pulp.value(prots[proteins])
            if proteins in protein_finder and parsimony_value == 1:
                p_ind = protein_finder.index(proteins)
                protein_object = scored_proteins[p_ind]
                lead_protein_objects.append(protein_object)
                lead_protein_identifiers.append(protein_object.identifier)
            else:
                if parsimony_value == 1:
                    # Why are some proteins not being found when we run exclusion???
                    logger.warning("Protein {} not found with protein finder...".format(proteins))
                else:
                    pass

        self.lead_protein_objects = lead_protein_objects

        grouped_proteins = self._create_protein_groups(
            all_scored_proteins=scored_data,
            lead_protein_objects=self.lead_protein_objects,
            grouping_type=self.data.parameter_file_object.grouping_type,
        )

        regrouped_proteins = self._swissprot_and_isoform_override(
            scored_data=scored_data,
            grouped_proteins=grouped_proteins,
            override_type="soft",
            isoform_override=True,
        )

        grouped_protein_objects = regrouped_proteins["grouped_protein_objects"]
        protein_group_objects = regrouped_proteins["group_objects"]

        # Get the higher or lower variable
        hl = self.data.higher_or_lower()

        logger.info("Sorting Results based on lead Protein Score")
        grouped_protein_objects = datastore.DataStore.sort_protein_objects(
            grouped_protein_objects=grouped_protein_objects, higher_or_lower=hl
        )
        protein_group_objects = datastore.DataStore.sort_protein_group_objects(
            protein_group_objects=protein_group_objects, higher_or_lower=hl
        )

        # Run lead reassignment for the group objets and protein objects
        protein_group_objects = self._reassign_protein_group_leads(
            protein_group_objects=protein_group_objects,
        )

        grouped_protein_objects = self._reassign_protein_list_leads(grouped_protein_objects=grouped_protein_objects)

        logger.info("Re Sorting Results based on lead Protein Score")
        grouped_protein_objects = datastore.DataStore.sort_protein_objects(
            grouped_protein_objects=grouped_protein_objects, higher_or_lower=hl
        )
        protein_group_objects = datastore.DataStore.sort_protein_group_objects(
            protein_group_objects=protein_group_objects, higher_or_lower=hl
        )

        self.data.grouped_scored_proteins = grouped_protein_objects
        self.data.protein_group_objects = protein_group_objects

    def infer_proteins(self):
        """
        This method performs the Parsimony inference method and uses pulp for the LP solver.

        This method assigns the variables: `grouped_scored_proteins` and `protein_group_objects`.
        These are both variables of the [DataStore object][pyproteininference.datastore.DataStore] and are
        lists of [Protein][pyproteininference.physical.Protein] objects
        and [ProteinGroup][pyproteininference.physical.ProteinGroup] objects.

        """

        if self.parameter_file_object.lp_solver == self.PULP:

            self._pulp_grouper()

        else:
            raise ValueError(
                "Parsimony cannot run if lp_solver parameter value is not one of the following: {}".format(
                    ", ".join(Inference.LP_SOLVERS)
                )
            )

        # Call assign shared peptides
        self._assign_shared_peptides(shared_pep_type=self.parameter_file_object.shared_peptides)

    def _assign_shared_peptides(self, shared_pep_type="all"):

        if not self.data.grouped_scored_proteins and self.data.protein_group_objects:
            raise ValueError(
                "Grouped Protein objects could not be found. Please run 'infer_proteins' method of the Parsimony class"
            )

        if shared_pep_type == self.ALL_SHARED_PEPTIDES:
            pass

        elif shared_pep_type == self.BEST_SHARED_PEPTIDES:
            logger.info("Assigning Shared Peptides from Parsimony to the Best Scoring Protein")
            raw_peptide_tracker = set()
            peptide_tracker = set()
            for prots in self.data.grouped_scored_proteins:
                new_psms = []
                new_raw_peptides = set()
                new_peptides = set()
                lead_prot = prots[0]
                for psm in lead_prot.psms:
                    raw_pep = psm.identifier
                    pep = psm.non_flanking_peptide
                    if raw_pep not in raw_peptide_tracker:
                        new_raw_peptides.add(raw_pep)
                        raw_peptide_tracker.add(raw_pep)
                    if pep not in peptide_tracker:
                        new_peptides.add(pep)
                        new_psms.append(psm)
                        peptide_tracker.add(pep)
                lead_prot.psms = new_psms
                lead_prot.raw_peptides = new_raw_peptides
                lead_prot.peptides = new_peptides

            raw_peptide_tracker = set()
            peptide_tracker = set()
            for group in self.data.protein_group_objects:
                lead_prot = group.proteins[0]
                new_psms = []
                new_raw_peptides = set()
                new_peptides = set()
                for psm in lead_prot.psms:
                    raw_pep = psm.identifier
                    pep = psm.non_flanking_peptide
                    if raw_pep not in raw_peptide_tracker:
                        new_raw_peptides.add(raw_pep)
                        raw_peptide_tracker.add(raw_pep)
                    if pep not in peptide_tracker:
                        new_peptides.add(pep)
                        new_psms.append(psm)
                        peptide_tracker.add(pep)

                lead_prot.psms = new_psms
                lead_prot.raw_peptides = new_raw_peptides
                lead_prot.peptides = new_peptides

        else:
            pass


class FirstProtein(Inference):
    """
    FirstProtein Inference class. This class contains methods that support the initialization of a
    FirstProtein inference method.

    Attributes:
        data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
        digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].

    """

    def __init__(self, data, digest):
        """
        FirstProtein Inference initialization method.

        Args:
            data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
            digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].

        Returns:
            object:
        """
        self.data = data
        self.digest = digest
        self.data._validate_scored_proteins()
        self.scored_data = self.data.get_protein_data()
        self.data = data

    def infer_proteins(self):
        """
        This method performs the First Protein inference method.

        This method assigns the variables: `grouped_scored_proteins` and `protein_group_objects`.
        These are both variables of the [DataStore object][pyproteininference.datastore.DataStore] and are
        lists of [Protein][pyproteininference.physical.Protein] objects
        and [ProteinGroup][pyproteininference.physical.ProteinGroup] objects.

        """

        grouped_proteins = self._create_protein_groups(scored_proteins=self.scored_data)

        # Get the higher or lower variable
        hl = self.data.higher_or_lower()

        logger.info("Applying Group ID's for the First Protein Method")
        regrouped_proteins = self._apply_protein_group_ids(
            grouped_protein_objects=grouped_proteins,
        )

        grouped_protein_objects = regrouped_proteins["grouped_protein_objects"]
        protein_group_objects = regrouped_proteins["group_objects"]

        logger.info("Sorting Results based on lead Protein Score")
        grouped_protein_objects = datastore.DataStore.sort_protein_objects(
            grouped_protein_objects=grouped_protein_objects, higher_or_lower=hl
        )
        protein_group_objects = datastore.DataStore.sort_protein_group_objects(
            protein_group_objects=protein_group_objects, higher_or_lower=hl
        )

        self.data.grouped_scored_proteins = grouped_protein_objects
        self.data.protein_group_objects = protein_group_objects


class PeptideCentric(Inference):
    """
    PeptideCentric Inference class. This class contains methods that support the initialization of a
    PeptideCentric inference method.

    Attributes:
        data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
        digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].

    """

    def __init__(self, data, digest):
        """
        PeptideCentric Inference initialization method.

        Args:
            data (DataStore): [DataStore Object][pyproteininference.datastore.DataStore].
            digest (Digest): [Digest Object][pyproteininference.in_silico_digest.Digest].

        Returns:
            object:
        """
        self.data = data
        self.digest = digest
        self.data._validate_scored_proteins()
        self.scored_data = self.data.get_protein_data()

    def infer_proteins(self):
        """
        This method performs the Peptide Centric inference method.

        This method assigns the variables: `grouped_scored_proteins` and `protein_group_objects`.
        These are both variables of the [DataStore object][pyproteininference.datastore.DataStore] and are
        lists of [Protein][pyproteininference.physical.Protein] objects
        and [ProteinGroup][pyproteininference.physical.ProteinGroup] objects.

        Returns:
            None:

        """

        # Get the higher or lower variable
        hl = self.data.higher_or_lower()

        logger.info("Applying Group ID's for the Peptide Centric Method")
        regrouped_proteins = self._apply_protein_group_ids()

        grouped_protein_objects = regrouped_proteins["grouped_protein_objects"]
        protein_group_objects = regrouped_proteins["group_objects"]

        logger.info("Sorting Results based on lead Protein Score")
        grouped_protein_objects = datastore.DataStore.sort_protein_objects(
            grouped_protein_objects=grouped_protein_objects, higher_or_lower=hl
        )
        protein_group_objects = datastore.DataStore.sort_protein_group_objects(
            protein_group_objects=protein_group_objects, higher_or_lower=hl
        )

        self.data.grouped_scored_proteins = grouped_protein_objects
        self.data.protein_group_objects = protein_group_objects

    def _apply_protein_group_ids(self):
        """
        This method creates the ProteinGroup objects for the peptide_centric inference based on protein groups
        from [._create_protein_groups][pyproteininference.inference.Inference._create_protein_groups].

        Returns:
            dict: a Dictionary that contains a list of [ProteinGroup]]pyproteininference.physical.ProteinGroup]
            objects (key:"group_objects") and a list of grouped [Protein]]pyproteininference.physical.Protein]
            objects (key:"grouped_protein_objects").

        """

        grouped_protein_objects = self.data.get_protein_data()

        # Here we create group ID's
        group_id = 0
        list_of_proteins_grouped = []
        protein_group_objects = []
        for protein_group in grouped_protein_objects:
            protein_group.peptides = set(
                [Psm.split_peptide(peptide_string=x) for x in list(protein_group.raw_peptides)]
            )
            protein_list = []
            group_id = group_id + 1
            pg = ProteinGroup(group_id)
            logger.debug("Created Protein Group with ID: {}".format(str(group_id)))
            # The following loop assigns group_id's, reviewed/unreviewed status, and number of unique peptides...
            if group_id not in protein_group.group_identification:
                protein_group.group_identification.add(group_id)
            protein_group.num_peptides = len(protein_group.peptides)
            # Here append the number of unique peptides... so we can use this as secondary sorting...
            protein_list.append(protein_group)
            # Sorted protein_groups then becomes a list of lists... of protein objects

            pg.proteins = protein_list
            protein_group_objects.append(pg)
            list_of_proteins_grouped.append([protein_group])

        return_dict = {
            "grouped_protein_objects": list_of_proteins_grouped,
            "group_objects": protein_group_objects,
        }

        return return_dict
