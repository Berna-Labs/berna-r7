# Berna R7 Architecture

**Version:** 1.0
**Lineage:** R5 (DNA-Kernel Plexus, DOI: 10.5281/zenodo.23015308)
**Model:** 1.06B parameters
**Trained from:** scratch (no fine-tuning, no distillation)
**Hardware:** 1× RTX 5090 (32 GB)

## Six-Layer Structure

- Sensors: tokenizer (32k BPE, En+Math+Code)
- Spinal Cord: rule-based router (deferred)
- Plexus: dynamic graph P = (V, E, W)
- Knowledge Cells: 112 cells, born/split/die
- DNA Kernel: 100 chromosomes x 4 genes
- Registry: SQL + UUIDs + connection tokens

## Cells

A cell = a group of neurons in one MLP layer.

- Partition: 4 cells per layer, 28 layers = 112 cells
- ID: <model_uuid>-cell-<layer><index>
- States: active / dormant / frozen / dead / split

## 6D Knowledge Vector

K = [K1..K6]:
- K1 (L) sequential coverage
- K2 (W) domain breadth
- K3 (H) abstraction
- K4 (D) reasoning depth
- K5 (T) temporal freshness
- K6 (E) encompassment

Saturation: S = ||K||_omega

## Constants (R5)

- S* = 0.85 (split threshold)
- T_cons = 3 (consecutive steps)
- eta_e = 0.01 (plexus LR)
- lambda_e = 0.001 (plexus decay)

## Per-Model File Layout

- manifest.json: uuid, name, parent, children, connection_token
- config.json: architecture
- model.safetensors: weights
- cells.json: cell registry
- plexus.json: graph state
- dna/: per-cell DNA files

## Code Modules (core/code/cells/)

- saturation.py: K vector + S
- dna_kernel.py: 100x4 chromosomes
- cell_split.py: H1 splitting
- cell_birth.py: initial partition
- plexus.py: dynamic graph + K4
- cell_manager.py: orchestrator
- facets.json: K6 taxonomy
