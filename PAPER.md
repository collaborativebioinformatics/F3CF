# Introduction

Team 8 was tasked with developing a meta-meta-graph of phenotype–genotype metagraphs derived from heterogeneous datasets. This framing raised a broader question: can phenotype–genotype knowledge be represented in a way that increases the utility of data across sources? The potential technical and clinical applications of such a representation were less clearly defined. In a Nordic biobank and Scandinavian healthcare context, the challenge is not only to integrate heterogeneous datasets, but also to determine which relationships are meaningful across data sources, phenotypes, genotypes, treatments, and patient populations. Although biobank-scale datasets are increasingly used for model development, translation of such models into routine clinical settings remains limited.[Andreoli L, et al. Polygenic risk scores in the clinic: a systematic review of stakeholders’ perspectives, attitudes, and experiences. European Journal of Human Genetics. 2025;33:266–280] A meta-meta-graph alone does not provide an explicit mechanism for quantifying data utility or for comparing heterogeneous relational structures.

We therefore propose F3CF, a Flexible Federated Framework for Multi-Relational Collaborative Factorization. F3CF is designed to learn a shared latent representation from distributed relational data, enabling the joint exploration of structure across phenotypes, genotypes, drugs, patients, or other entities. The framework is designed to be extensible, allowing new data sources, entity types, and relations to be incorporated over time. A federated implementation further enables patient-level relations to contribute to model training without requiring patient-level data to leave the originating institution, while shared non-patient parameters can be updated collaboratively across sites.

In this work, we present a proof-of-concept implementation of F3CF and evaluate its potential for shared latent-space construction, cross-source relational analysis, and assessment of data-source utility.

# Methods

The main F3CF benchmark combined synthetic UK Biobank-style EHR data with genetic information derived from the PGS Catalog. Clinical data were converted upstream into site-specific patient × phenotype matrices, while PGS Catalog scoring files and trait mappings were used to define the corresponding genetic feature space. Clinical and genetic features were aligned through shared phenotype identifiers and distributed across three simulated sites with differing cohort sizes, partially overlapping phenotype coverage, and incomplete availability of genomic data. The current F3CF implementation operates on these prepared relation matrices rather than performing the underlying EHR phenotyping directly.

A separate synthetic benchmark was used to evaluate federated PRS construction. Six quantitative traits were simulated with known causal loci and heritabilities across cohorts of increasing combined sample size. This provided a defined ground truth against which recovery of locus–trait associations and PRS performance could be assessed as additional sites were incorporated.
Federated collaborative factorization was implemented using NVIDIA FLARE. For each relation rr, the site-specific matrix was approximated as

$$X_k^(r) ≈ U_k^(r) V^(r)T$$

where $U_k^(r)$ contains local row embeddings and $V^(r)$ contains shared phenotype embeddings. Patient-level data and row embeddings remained local. Binary patient–phenotype relations were optimized with binary cross-entropy, while continuous genetic relations used mean-squared reconstruction error. Shared phenotype embeddings were aggregated by size-weighted federated averaging, with only sites observing a given phenotype contributing to its update, followed by $L_2$-normalization.

flowchart TB

    %% =========================
    %% DISTRIBUTED SITES
    %% =========================
    subgraph SITES["Distributed sites"]
        direction LR

        subgraph SA["Biobank A"]
            direction TB
            A["Patient × Phenotype<br/>Patient × PRS"]
            LA["Local factorization"]
            A --> LA
        end

        subgraph SB["Biobank B"]
            direction TB
            B["Patient × Phenotype<br/>Patient × PRS"]
            LB["Local factorization"]
            B --> LB
        end

        subgraph SC["Clinic C"]
            direction TB
            C["Patient × Phenotype"]
            LC["Local factorization"]
            C --> LC
        end
    end

    %% =========================
    %% PRIVACY
    %% =========================
    P["Patient-level data and patient embeddings remain local"]

    %% =========================
    %% FEDERATED CORE
    %% =========================
    AGG["Federated aggregation + redistribution<br/>NVIDIA FLARE"]

    SH["Shared phenotype embeddings"]

    LAT["Shared latent space"]

    %% =========================
    %% CONNECTIONS
    %% =========================
    LA -->|"shared parameter updates"| AGG
    LB --> AGG
    LC --> AGG

    AGG --> SH
    SH --> LAT

    LA -.-> P
    LB -.-> P
    LC -.-> P

    %% =========================
    %% STYLING
    %% =========================
    classDef input fill:#F4F4F4,stroke:#777,stroke-width:1.2px,color:#222;
    classDef local fill:#DCEBFA,stroke:#2867B2,stroke-width:1.4px,color:#111;
    classDef core fill:#FFFFFF,stroke:#666,stroke-width:1.8px,color:#111;
    classDef shared fill:#DDD3EC,stroke:#8878A5,stroke-width:1.5px,color:#111;
    classDef latent fill:#E8F2FF,stroke:#3478D4,stroke-width:1.6px,color:#111;
    classDef note fill:#FFFBEA,stroke:#C9A227,stroke-width:1.2px,color:#333;

    class A,B,C input;
    class LA,LB,LC local;
    class AGG core;
    class SH shared;
    class LAT latent;
    class P note;

The prototype used 32 latent factors, five federated rounds, 20 local epochs per round, and a learning rate of 0.05. Clinical and genetic relations were evaluated both separately and in a coupled phenotype space.
Clinical patient–phenotype observations were modeled using binary cross-entropy, whereas continuous genetic relations were optimized using mean-squared reconstruction error. The prototype used 32 latent factors, five federated communication rounds, 20 local epochs per round, and a learning rate of 0.05. Site contributions were aggregated using size-weighted federated averaging, with updates restricted to phenotypes observed at the contributing site. Sites without genetic information contributed only to the clinical representation. Clinical and genetic relations were examined both as separate latent representations and in a coupled phenotype space, allowing shared and relation-specific structure to be compared.

Federated reconstruction was compared with training on pooled data as a centralized reference. For predictive evaluation, 10% of patient–phenotype observations were held out at each site and scored using area under the receiver-operating characteristic curve (AUC). Simple prevalence-based and PRS-informed models, together with randomly initialized embeddings, were used as additional reference models.

Latent phenotype structure was evaluated from pairwise cosine similarities between phenotype embeddings. Similarity networks were constructed using a cosine-similarity threshold of 0.65, and clinical and genetic networks were compared by identifying shared, clinical-only, and genetic-only edges. For the federated PRS benchmark, increasing numbers of sites were combined and evaluated against the known simulated genetic architecture. Performance was quantified by recovery of true and false locus–trait associations and by trait-specific PRS R^2.

# Results

Federated training closely reproduced centralized training on the synthetic benchmark, suggesting that little performance was lost through federation itself. Simple prevalence-based and PRS-informed baselines performed better, indicating that the current factorization primarily demonstrates the feasibility of the federated implementation. The learned genetic phenotype representation nevertheless showed non-random agreement with the reference latent structure. 

Without coupling, the clinical and genetic relations produced distinct phenotype similarity networks (Fig. 1). Both latent spaces showed clustering by broad phenotype category, but their topology differed substantially. The genetic representation was denser, with 27,677 edges above the similarity threshold compared with 17,353 in the non-genetic representation.

[Fig. 1](images\coupled_latent_space_edge_comparison.png)

With coupling, the two relations were represented in a common phenotype space (Fig. 2). At the selected similarity threshold, 13,728 edges were shared between the clinical and genetic networks, while 3,445 were clinical-only and 13,949 genetic-only. The shared and relation-specific edges show that common latent structure could be retained while allowing the two data sources to encode distinct associations.

[Fig. 2](images\coupled_latent_space_edge_comparison.png)

In an experimental implementation of federated PRS construction, increasing the number of contributing sites markedly improved recovery of the simulated locus–trait structure (Fig. 3). With one site (N=396N=396), 91 associations were detected, of which 9 were causal and 82 were false positives. At three sites (N=2,398N=2,398), 44 of 54 detected associations were causal, and by six sites (N=8,000N=8,000), all 123 detected associations corresponded to simulated causal loci. Recovery was trait-dependent, with loci for height detected earlier than those for weaker traits such as BMI and heart rate. 

[Fig. 3](images\trait_curves.png)

This increase in recovered genetic signal was accompanied by improved PRS performance (Fig. 4). PRS R2R^2 increased as additional sites were incorporated for all six traits. Height and HbA1c showed the largest gains, reaching approximately 0.45 at six sites, while body weight and HDL cholesterol reached approximately 0.27 and 0.22, respectively. BMI and heart rate showed smaller gains. Overall, increasing the amount of federated data improved both recovery of the underlying locus–trait structure and the variance explained by the resulting polygenic scores. 

[Fig. 4](images\metagraph_bipartite.png)

# Conclusion
* F3CF demonstrated that collaborative factorization can be performed across distributed sites while keeping patient-level data and embeddings local.
* The learned latent space retained both shared and source-specific clinical/genetic structure, supporting integration of heterogeneous data without requiring identical relationships across sources.
* In the federated PRS benchmark, adding more sites improved recovery of causal locus–trait associations and increased PRS R^2, illustrating the value of combining distributed evidence.
