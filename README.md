# F3CF: A Flexible Federated Framework for Multi-Relational Collaborative Factorization

F3CF is a framework for federated exploration of a shared, multi-relational, multi-institutional latent knowledge space.

## Demo

Insert demo here.

## Mission

## Infographics

Choose one of:

* [Infographic 1](images/infograph1.png)
* [Infographic 2](images/infograph2.png)
* [Infographic 3](images/infograph3.png)
* [Infographic 4](images/infograph4.png)

## Milestones

## Flowchart

```mermaid
flowchart TB

    %% =========================
    %% TOP ROW — Institutions
    %% =========================
    subgraph INST["Distributed Data Sources"]
        direction LR

        subgraph C1["Clinic A"]
            direction TB
            A1[/Patient × Drug/]
            A2[/Patient × Phenotype/]
            A3[/Optional local relations/]
        end

        subgraph C2["Clinic B"]
            direction TB
            B1[/Patient × Drug/]
            B2[/Patient × Phenotype/]
            B3[/Optional local relations/]
        end

        subgraph BB["Biobank"]
            direction TB
            G1[\Phenotype × PRS\]
            G2[\Optional genomic relations\]
        end
    end

    %% =========================
    %% CORE
    %% =========================
    F3CF(["F3CF<br/>Flexible Federated Framework for<br/>Multi-Relational Collaborative Factorization"])

    P{{Raw data stay local<br/>Only model parameters / updates are shared}}

    %% =========================
    %% REPRESENTATIONS
    %% =========================
    subgraph REP["Learned Representations"]
        direction LR
        L[[Local embeddings<br/>Site-specific patients]]
        S[[Shared embeddings<br/>Phenotypes · Drugs · PRS]]
    end

    Z[(Shared N-dimensional<br/>latent space)]

    %% =========================
    %% OUTPUTS
    %% =========================
    subgraph OUT["Exploration & Applications"]
        direction LR
        O1>Patient stratification]
        O2>Drug response]
        O3>Genotype–phenotype discovery]
        O4>Biobank utility]
        O5>Latent structure exploration]
    end

    %% =========================
    %% CONNECTIONS
    %% =========================
    A1 -->|local relation| F3CF
    A2 -->|local relation| F3CF
    A3 -.->|optional| F3CF

    B1 -->|local relation| F3CF
    B2 -->|local relation| F3CF
    B3 -.->|optional| F3CF

    G1 -->|genomic relation| F3CF
    G2 -.->|optional| F3CF

    P -.->|federated learning| F3CF

    F3CF -->|site-specific| L
    F3CF -->|shared across sites| S

    L --> Z
    S --> Z

    Z --> O1
    Z --> O2
    Z --> O3
    Z --> O4
    Z --> O5

    %% =========================
    %% STYLING
    %% =========================
    classDef clinic fill:#EAF4FF,stroke:#4A90E2,stroke-width:1.5px,color:#123;
    classDef biobank fill:#E8FFF6,stroke:#20B27A,stroke-width:1.5px,color:#123;
    classDef core fill:#F3E8FF,stroke:#8E44AD,stroke-width:2.5px,color:#123;
    classDef embedding fill:#EAFBF3,stroke:#2E8B57,stroke-width:1.5px,color:#123;
    classDef latent fill:#FFF0F7,stroke:#D63384,stroke-width:2.5px,color:#123;
    classDef output fill:#F5F5F5,stroke:#666,stroke-width:1.2px,color:#123;
    classDef note fill:#FFFBEA,stroke:#C9A227,stroke-width:1.2px,color:#123;

    class A1,A2,A3,B1,B2,B3 clinic;
    class G1,G2 biobank;
    class F3CF core;
    class L,S embedding;
    class Z latent;
    class O1,O2,O3,O4,O5 output;
    class P note;
```

## How it works

F3CF represents distributed clinical and biobank data as a set of related matrices, such as patient–drug, patient–phenotype, and phenotype–genotype (e.g., PRS) relationships. Each site trains locally on the relations it holds, while shared entity representations are updated collaboratively across sites.

The framework learns a common N-dimensional latent space in which patients, phenotypes, drugs, PRS, and other entities can be compared and clustered. Patient-level representations can remain site-specific, while shared entities such as phenotypes and drugs are aligned across institutions.

Because the model is relational and modular, new clinics, entities, columns, or relation types can be added without redesigning the entire system. Raw data remain local; only model parameters or updates are exchanged during federated training.

The resulting latent space can be explored for tasks such as patient stratification, drug-response prediction, genotype–phenotype discovery, and estimating whether external biobank data add useful information to a specific clinic.

Each data source is represented as a relation matrix, for example patient–drug, patient–phenotype, or phenotype–PRS.

F3CF learns low-dimensional embeddings such that:

$$
R^{(r)} \approx Z_a W_r Z_b^\top
$$

where $Z_a$ and $Z_b$ are latent representations of the connected entities, and $W_r$ captures relation-specific structure.

The model jointly optimizes all available relations, sharing common embeddings across sites while keeping site-specific patient representations local.

## Synthetic datsets

For implementation testing we generate patient-level data.

## Example F3CF using synthetic datasets

Here we show results from using out implementation on our synthetic datasets.

## Exploring the latent space

F3CF learns a shared latent structure for globally aligned entities such as phenotypes, drugs, and genomic features, while patient representations remain local to each institution.

This enables several forms of exploration:

* **Patient-level exploration** — a local patient representation can be queried against the shared model to identify associated phenotypes, genetic-risk features, and drugs without sharing the patient embedding across sites.
* **Phenotype-level exploration** — examine which drugs, genomic features, and local patient profiles are most strongly associated with a phenotype.
* **Drug-level exploration** — investigate which phenotypic or genomic patterns are associated with a drug or treatment response.
* **Population-level exploration** — within each site, compare or cluster local patient representations relative to the globally learned latent structure.

The latent space can also be projected into two or three dimensions for visualization, while relation-specific similarities are evaluated in the full learned space.

## Exploring data source relationships

Each source can be treated as a set of observations that contributes to the shared model. Its contribution can then be assessed indirectly by measuring how the learned structure or downstream performance changes when the source is added, removed, or perturbed.

For a data source $D_k$, its structural contribution can be measured by comparing models trained with and without that source:

$$
S(D_k) = d(Z_{all}, Z_{without\ D_k})
$$

where $d(\cdot,\cdot)$ is an alignment-aware measure of change in the latent structure, such as changes in pairwise similarities, nearest-neighbour structure, or clustering.

For a target clinic $C$, the utility of an external data source can be defined as:

$$
U(D_k \rightarrow C) = Performance(C \mid D_k) - Performance(C)
$$

This measures whether including information from $D_k$ improves performance on a defined task at clinic $C$.

## Results from the synthetic data



## Future aspects

Firstly, F3CF should be tested on representative real-world datasets to determine whether the learned latent space provides clinically meaningful utility. This includes evaluating whether it recovers known biological relationships, improves prediction or stratification, and whether external data sources add measurable value to a clinical site. 

Rather than limiting F3CF to classical matrix factorization, the relation operator $W_r$ could be modular. It could be a matrix factorization operator, a knowledge-graph embedding such as a bilinear relation, or potentially a graph-neural-network component. Different relations could then use different mathematical models while still contributing to the same shared representation.

Relationships could also be extended using known relational graphs, ontologies, and other structured knowledge. For example, phenotype ontologies, gene–pathway relationships, drug–target interactions, and disease–gene associations could provide additional constraints on the latent space. This would allow F3CF to combine relationships learned from distributed data with established biological knowledge.

Such extensions could turn F3CF into a framework for exploring a federated **meta-knowledge graph**, where clinical observations, genomic associations, treatments, phenotypes, and existing biomedical knowledge contribute to a common latent representation. This could enable exploration of relationships that are not directly observed in any single dataset while preserving the distributed nature of the underlying data.

## Team 8 — Nordic Conference on Future Health 2026, 14–16 September 2026

* Victor Enrique Goitea
* Chris Hart
* Davor Vukadin
* Henrik Formoe
* Edvin Smajlovic
* Sebastian Krog [writer]
* Elakiya Sivakumar
