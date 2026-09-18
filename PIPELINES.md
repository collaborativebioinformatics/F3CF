# F3CF pipelines

This document separates the pipelines that are implemented in the current proof of concept from those that are experimental or expected extensions of F3CF.

## Overview

The intended data flow is:

```text
Raw local data
    ↓
Source-specific preprocessing
    ↓
Standardized relation matrices
    ↓
Local multi-relational factorization
    ↓
Federated aggregation of selected shared entities
    ↓
Shared latent structure
    ↓
Local or cross-source exploration and analyses
```

Patient-level observations and patient embeddings remain local to the originating institution.
