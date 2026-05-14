# StructuralVariantEngine

Pure Python structural variant detection and analysis pipeline.

## Features
- SV detection from read-pair signatures (DEL, DUP, INV, TRA, INS)
- Breakpoint refinement using split-read evidence
- SV genotyping (HET/HOM from VAF)
- Cancer driver gene disruption analysis
- BCR-ABL1 translocation detection

## Usage
```bash
pip install numpy scipy matplotlib
python structural_variant_engine.py
```

## Results (10 tumor-normal pairs, 1500 true SVs)
- Precision=0.923, Recall=0.313, F1=0.467
- BCR-ABL1 translocation: 1 candidate
