# F3CF: A Flexible Federated Framework for Multi-Relational Collaborative Factorization
F3CF is a framework for federated exploration of a shared multi-relational, multi-institutional latent knowledge space.
Nordic Conference on Future Health 2026 (14–16 September 2026).

## DEMO

Insert demo here.

## Mission

## Infographs

Choose one of:
- [Infograph 1](images/infograph1.png)
- [Infograph 2](images/infograph2.png)
- [Infograph 3](images/infograph3.png)
- [Infograph 4](images/infograph4.png)

## Milestones


## Flowchart
This version forces the institutions into a top row, left to right, with F3CF underneath.

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

## Step 1
Phenotypes x PRS (Genetic libality), Phenotypes x patient, Drug x Patient
2 synthetic datasources: 1 clinic and 1 biobank
Produce a relevant model.

## Later steps...

## Step N

Use this on real data.

## Team 8
* Victor Enrique Goitea
* Chris Hart
* Davor Vukadin
* Henrik Formoe
* Edvin Smajlovic
* Sebastian Krog [writer]
* Elakiya Sivakumar


