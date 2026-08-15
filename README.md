# TargetSA

Official implementation of **TargetSA: adaptive simulated annealing for target-specific drug design** (Bioinformatics, 2025).

TargetSA performs target-specific molecular generation and multi-objective optimization through adaptive simulated annealing. It combines a history-guided GNN position predictor, four molecular graph editing operations (insertion, replacement, deletion, and cyclization), and a reversible sampling strategy to optimize docking affinity, drug-likeness, and synthetic accessibility.

- Paper: [Bioinformatics](https://doi.org/10.1093/bioinformatics/btae730)

## Method overview

Starting from the 112 drug-like fragments in `frag112.txt`, TargetSA:

1. predicts promising editing positions with a pretrained GNN;
2. generates candidates through molecular graph editing;
3. evaluates candidates with docking and molecular-property objectives;
4. accepts or rejects candidates according to the simulated-annealing temperature;
5. applies reversible sampling before finally rejecting a candidate.

## Datasets

The paper reports experiments on two public datasets:

- **CrossDocked2020**: 100,000 protein-ligand complexes for training and 100 pockets for testing. AutoDock Vina is used for this benchmark.
  - [CrossDocked2020 v1.1 download](https://bits.csb.pitt.edu/files/crossdock2020/v1.1/)
  - [Pocket2Mol processed subset and split](https://github.com/pengxingang/Pocket2Mol/tree/main/data)
- **Binding MOAD**: 130 test pockets prepared following DiffSBDD. QuickVina 2 is used for this benchmark.
  - [Preprocessed CrossDocked2020 and Binding MOAD datasets](https://doi.org/10.5281/zenodo.13931612)
  - [DiffSBDD data preparation instructions](https://github.com/arneschneuing/DiffSBDD#benchmarks)

Each target pocket should contain a receptor structure and its reference ligand.

## Environment

The released environment was built on Linux with:

- Python 3.8.15
- PyTorch 1.13.1 and CUDA 11.6
- PyTorch Geometric 2.4.0
- RDKit 2023.9.1
- Open Babel 3.1.1
- Meeko 0.5.0
- AutoDockTools and PDB2PQR
- QuickVina 2

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/XueZhe-Zachary/TargetSA.git
cd TargetSA

conda env create -f env.yaml
conda activate TargetSA
```

For the Binding MOAD workflow, install [QuickVina 2](https://qvina.github.io/compilingQvina2.html) and make sure its executable is available as `qvina2.1`:

```bash
command -v obabel
command -v qvina2.1
```

## Input preparation

Create one directory for each target pocket:

```text
example_pocket/
├── receptor.pdb
├── reference_ligand.sdf
└── prepared_rep.pdbqt    # optional
```

- The SDF ligand must contain valid 3D coordinates. Its centroid is used as the docking-box center.
- If `prepared_rep.pdbqt` is not provided, the receptor is prepared from `receptor.pdb` with PDB2PQR and AutoDockTools.
- Keep only one `.pdb` receptor and one `.sdf` reference ligand in each pocket directory.

## Run TargetSA

Run the following command from the repository root:

```bash
python SA_new.py --pocket_path /absolute/path/to/example_pocket
```

Main parameters used by the released implementation:

| Parameter | Value |
| --- | ---: |
| Random seed | `1000` |
| Initial temperature | `1.0` |
| Minimum temperature | `0.1` |
| Editing attempts per temperature | `5` |
| Top-K editing positions | `5` |
| Similarity threshold | `0.1` |
| Docking box size | `10 x 10 x 10 Å` |
| Docking exhaustiveness | `8` |

The runner optimizes all 112 fragments in `frag112.txt` for the specified pocket.

## Output

Results are saved as `results.pkl` inside the pocket directory. The file contains:

- initial and optimized RDKit molecules;
- initial and optimized docking scores;
- generated molecules with docking coordinates;
- starting fragments that failed during generation.

## Citation

```bibtex
@article{xue2025targetsa,
  title   = {TargetSA: adaptive simulated annealing for target-specific drug design},
  author  = {Xue, Zhe and Sun, Chenwei and Zheng, Wenhao and Lv, Jiancheng and Liu, Xianggen},
  journal = {Bioinformatics},
  volume  = {41},
  number  = {1},
  pages   = {btae730},
  year    = {2025},
  doi     = {10.1093/bioinformatics/btae730}
}
```
