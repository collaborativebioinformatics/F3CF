```mermaid
graph TD;
    A{"Data Acquisition: Odyssey EHR, UKB synth"}-->B{"Gefion Setup: Gentype-Phenotype data"};
    B-->C{"Model Data: Enrich phenotype information from EHRs"};
    C-->D{"Explore OMOP with Team5"};
    C-->E{"Use Collaborative Filtering to develop associations"};
    E-->G{"Lightweight search frontend for phenotypes"}
    G-->F{"Demonstrate Variation in genotype associations - simple example, dissimilar subphenotypes"};
```
