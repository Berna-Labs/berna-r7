# Berna DNA Kernel

**Version:** 1.0
**Source:** R5 Section 3.5
**Location:** core/code/cells/dna_kernel.py

## Structure

Each cell carries DNA:
D_c = { (a_i, s_i, p_i, pi_i) } for i = 1..100

Genes:
- a  = activation
- s  = sensitivity
- p  = persistence
- pi = plasticity

All genes are floats in [0, 1].

## The 10 Chromosome Groups

100 chromosomes = 10 groups x 10 chromosomes:

- language
- math
- code
- science
- sensory
- reasoning
- memory
- coordination
- adaptation
- growth

## Control Signals

The kernel maps (D_c, K_c, u_c) -> {route, grow, split, retain}:

- route  = mean(a)
- grow   = mean(pi) - mean(p) + 0.5   (threshold > 0.6)
- split  = active if S >= S*
- retain = mean(p) - mean(pi) + 0.5   (threshold > 0.7)

Decision priority (R5):
1. S >= S* -> split
2. retain > 0.7 -> retain
3. grow > 0.6 -> grow
4. else -> route

## Mutations

On split:
- each child copies parent DNA
- each gene mutated: g' = clamp(g + U(-r, r), 0, 1)
- default r = 0.02
- generation = parent + 1

## Signature

signature = SHA-256(gene matrix)[:32]
Used for integrity; changes on any mutation.

## Persistence

Stored at: {model_uuid}/dna/{cell_id}.dna.json

## Status

- Implemented in dna_kernel.py
- Tested: S=0.85 -> SPLIT, S=0.95 -> SPLIT
- Integration with live training: deferred until Mother finishes
