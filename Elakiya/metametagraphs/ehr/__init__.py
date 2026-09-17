"""EHR -> subphenotype definitions (README milestone 5).

Reads EHR data (UK Biobank tabular + GP records, or OMOP CDM tables), applies
a sourced, editable rule set (diagnosis codes, lab thresholds with units and
sex-specific cutoffs, medication evidence, minimum counts, time windows,
lab aggregation) over a subphenotype hierarchy, adds PheWAS phecodes, and
writes the output contract described in :mod:`metametagraphs.ehr.build`.

    python -m metametagraphs.ehr build --source toy --input /tmp/toy --out /tmp/subpheno
"""

from metametagraphs.ehr.build import OUTPUT_FILES, build  # noqa: F401
