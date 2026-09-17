# HPO excerpt

`hp_excerpt.obo` holds the `id`, `name` and `is_a` lines (unchanged text) of every term listed in
`../hpo_catalog.txt` and all of their ancestors, taken from

    https://raw.githubusercontent.com/obophenotype/human-phenotype-ontology/master/hp.obo
    data-version: hp/releases/2026-09-01 (downloaded 2026-09-17)

It lets the tests and the HPO export run offline. The full ontology is available from the URL above
(`metametagraphs.ehr.hpo.download_hpo`).

Attribution: this product uses the Human Phenotype Ontology (release 2026-09-01), created by the
Human Phenotype Ontology Consortium and the Monarch Initiative (Peter Robinson, Sebastian Koehler
et al.). License: see https://hpo.jax.org/app/license (the `terms:license` value in hp.obo); the
license page could not be retrieved from the build environment, so its redistribution terms were
not re-checked.

`../hpo_catalog.txt`: the 32 HPO ids of `data/ehr_lipids/phenotype_ids.txt` (PR #3, same order)
followed by the crosswalk terms that are not in that list (HP:0000112, HP:0012653, HP:0040217).
