# ARC-AGI-2 Solver

## Overview

This repository contains my research and development of a modular solver for ARC-AGI-2.

The project investigates individual grid properties and their relationships across training examples. Each investigation is organised into its own stage, allowing hypotheses to be tested, results recorded, and findings evaluated before being integrated into the solver.

## Research Website

**Website:** In development.

The website will provide a visual interface for exploring the solver's architecture, investigation stages, experimental results, and progress.

<!-- Replace the status above with a link to the deployed website when it is available. -->

## Solver Architecture

The solver is organised into eight investigation and solving stages.

| Stage | Purpose | Documentation |
|---|---|---|
| 1. Grid Sizes | Investigates input and output dimensions. | [Grid Sizes](solver/grid sizes/README.md) |
| 2. Grid Colours | Investigates colours added, removed, or maintained. | [Grid Colours](solver/grid colours/README.md) |
| 3. Grid Shapes | Investigates connected regions of the same colour. | [Grid Shapes](grid%20shapes/README.md) |
| 4. Grid Cells | Investigates individual grid cells and their properties. | [Grid Cells](grid%20cells/README.md) |
| 5. Grid Rows and Columns | Investigates row and column patterns and changes. | [Grid Rows and Columns](grid%20rows%20and%20columns/README.md) |
| 6. Grid Transformations | Investigates transformations between input and output grids. | [Grid Transformations](grid%20transformations/README.md) |
| 7. Grid Solver | Combines supported findings to generate solutions. | [Grid Solver](grid%20solver/README.md) |
| 8. Grid Verifier | Evaluates generated solutions and identifies failures. | [Grid Verifier](grid%20verifier/README.md) |

*Note: Documentation links will become active as the corresponding README files are created. Folder paths should match the actual repository structure.*

## Investigation Methodology

Each investigation stage is designed to examine particular grid properties independently.

The general research process is to:

1. Identify a property or propose a hypothesis.
2. Analyse the available training examples.
3. Record findings and compare predictions with available answers.
4. Investigate incorrect predictions and inconsistent patterns.
5. Determine whether the findings are sufficiently supported for integration into the solver.

A pattern observed in a limited number of examples is treated as a hypothesis until further testing provides evidence of its reliability.

## Experimental Results

Investigation results are stored alongside their corresponding stages, where applicable.

CSV files provide structured records of experimental findings and allow individual tasks to be inspected. Python scripts perform the analyses, while stage-specific README files document the methods, assumptions, and limitations.

## Current Status

The project is under active development.

The investigation stages may differ in their level of completion. A completed investigation does not automatically mean that its findings have been integrated into the solving pipeline or that the complete solver can solve the corresponding tasks.

## Repository Structure

Each investigation stage maintains its own documentation, Python scripts, and results.

The repository is organised to keep the following components distinct:

- **Documentation:** Explains the purpose and methodology of each stage.
- **Hypotheses:** Contains experiments used to investigate possible patterns.
- **Implementation:** Contains the Python code for each stage.
- **Results:** Contains recorded findings used for analysis and verification.

## License

See [LICENSE](LICENSE) for the terms governing the use of this repository.
