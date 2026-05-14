"""
StructuralVariantEngine: Structural Variant Detection and Analysis Pipeline
- SV detection from read-pair signatures (deletion, duplication, inversion, translocation)
- Breakpoint refinement using split-read evidence
- SV genotyping (homozygous/heterozygous)
- Cancer SV driver analysis
- SV annotation (gene disruption, fusion potential)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 60)
print("StructuralVariantEngine v1.0")
print("Structural Variant Detection and Analysis Pipeline")
print("=" * 60)

# ─── 1. GENOME SIMULATION ────────────────────────────────────
GENOME_SIZE = 3_000_000_000  # 3 Gb (human genome)
N_CHROMOSOMES = 23
CHROM_SIZES = np.array([
    248956422, 242193529, 198295559, 190214555, 181538259,
    170805979, 159345973, 145138636, 138394717, 133797422,
    135086622, 133275309, 114364328, 107043718, 101991189,
    90338345, 83257441, 80373285, 58617616, 64444167,
    46709983, 50818468, 156040895
])

print(f"\n[Genome] {N_CHROMOSOMES} chromosomes, {GENOME_SIZE/1e9:.1f} Gb")

# ─── 2. SIMULATE SEQUENCING DATA ─────────────────────────────
N_SAMPLES = 10  # tumor-normal pairs
N_TRUE_SVS = 150  # true SVs per sample

print(f"\n[Sequencing] Simulating {N_SAMPLES} tumor-normal pairs...")
print(f"  True SVs per sample: {N_TRUE_SVS}")

# SV types and frequencies (cancer-like)
SV_TYPES = ['DEL', 'DUP', 'INV', 'TRA', 'INS']
SV_FREQS = [0.45, 0.25, 0.15, 0.10, 0.05]

def simulate_sv(sv_type, chrom_sizes):
    """Simulate a single SV."""
    chrom1 = np.random.choice(len(chrom_sizes), p=chrom_sizes/chrom_sizes.sum())
    pos1 = np.random.randint(1000, chrom_sizes[chrom1] - 1000)

    if sv_type == 'TRA':
        chrom2 = np.random.choice(len(chrom_sizes), p=chrom_sizes/chrom_sizes.sum())
        while chrom2 == chrom1:
            chrom2 = np.random.choice(len(chrom_sizes), p=chrom_sizes/chrom_sizes.sum())
        pos2 = np.random.randint(1000, chrom_sizes[chrom2] - 1000)
        size = 0
    else:
        chrom2 = chrom1
        # SV size distribution (log-normal)
        size = int(np.random.lognormal(mean=8, sigma=2))  # median ~3kb
        size = max(50, min(size, 10_000_000))
        pos2 = min(pos1 + size, chrom_sizes[chrom1] - 1)

    return {
        'type': sv_type,
        'chrom1': chrom1 + 1,
        'pos1': pos1,
        'chrom2': chrom2 + 1,
        'pos2': pos2,
        'size': size,
    }

# Generate true SVs
all_true_svs = []
for s in range(N_SAMPLES):
    sample_svs = []
    for _ in range(N_TRUE_SVS):
        sv_type = np.random.choice(SV_TYPES, p=SV_FREQS)
        sv = simulate_sv(sv_type, CHROM_SIZES)
        sv['sample'] = s
        sv['vaf'] = np.random.beta(2, 3)  # variant allele frequency
        sv['genotype'] = 'HET' if sv['vaf'] < 0.7 else 'HOM'
        sample_svs.append(sv)
    all_true_svs.extend(sample_svs)

print(f"  Total true SVs: {len(all_true_svs)}")

# ─── 3. SV DETECTION FROM READ-PAIR SIGNATURES ───────────────
print("\n[Detection] Detecting SVs from read-pair signatures...")

def detect_svs_from_reads(true_svs, sensitivity=0.85, specificity=0.92, fp_rate=0.3):
    """
    Simulate SV detection with realistic sensitivity/specificity.
    Returns detected SVs with evidence scores.
    """
    detected = []
    for sv in true_svs:
        # Detection probability depends on SV size and type
        size_factor = min(1.0, np.log10(max(sv['size'], 100)) / 6)
        type_sensitivity = {'DEL': 0.90, 'DUP': 0.85, 'INV': 0.75, 'TRA': 0.80, 'INS': 0.60}
        p_detect = sensitivity * type_sensitivity[sv['type']] * (0.5 + 0.5 * size_factor)
        p_detect *= (0.5 + sv['vaf'])  # harder to detect low-VAF SVs

        if np.random.random() < p_detect:
            # Add noise to breakpoint positions
            noise = np.random.randint(-50, 50)
            # Evidence scores
            n_discordant = np.random.poisson(lam=max(5, sv['vaf'] * 30))
            n_split = np.random.poisson(lam=max(2, sv['vaf'] * 15))
            quality = min(60, n_discordant + n_split * 2)

            detected.append({
                **sv,
                'pos1_detected': sv['pos1'] + noise,
                'pos2_detected': sv['pos2'] + noise,
                'n_discordant': n_discordant,
                'n_split': n_split,
                'quality': quality,
                'true_positive': True,
            })

    # Add false positives
    n_fp = int(len(true_svs) * fp_rate)
    for _ in range(n_fp):
        sv_type = np.random.choice(SV_TYPES, p=SV_FREQS)
        fp_sv = simulate_sv(sv_type, CHROM_SIZES)
        fp_sv['sample'] = np.random.randint(N_SAMPLES)
        fp_sv['vaf'] = np.random.beta(1, 5)
        fp_sv['genotype'] = 'HET'
        fp_sv['pos1_detected'] = fp_sv['pos1']
        fp_sv['pos2_detected'] = fp_sv['pos2']
        fp_sv['n_discordant'] = np.random.poisson(lam=3)
        fp_sv['n_split'] = np.random.poisson(lam=1)
        fp_sv['quality'] = np.random.randint(5, 25)
        fp_sv['true_positive'] = False
        detected.append(fp_sv)

    return detected

detected_svs = detect_svs_from_reads(all_true_svs)
print(f"  Total detected SVs: {len(detected_svs)}")
print(f"  True positives: {sum(1 for s in detected_svs if s['true_positive'])}")
print(f"  False positives: {sum(1 for s in detected_svs if not s['true_positive'])}")

# ─── 4. QUALITY FILTERING ────────────────────────────────────
print("\n[Filtering] Applying quality filters...")

quality_threshold = 20
filtered_svs = [s for s in detected_svs if s['quality'] >= quality_threshold and
                s['n_discordant'] >= 3 and s['n_split'] >= 1]

tp_filtered = sum(1 for s in filtered_svs if s['true_positive'])
fp_filtered = sum(1 for s in filtered_svs if not s['true_positive'])
fn = len(all_true_svs) - tp_filtered

precision = tp_filtered / (tp_filtered + fp_filtered) if (tp_filtered + fp_filtered) > 0 else 0
recall = tp_filtered / len(all_true_svs) if len(all_true_svs) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"  After filtering (quality>={quality_threshold}):")
print(f"  Retained: {len(filtered_svs)} SVs")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  F1: {f1:.3f}")

# ─── 5. SV TYPE DISTRIBUTION ─────────────────────────────────
print("\n[Types] SV type distribution...")

type_counts = {}
for sv_type in SV_TYPES:
    n = sum(1 for s in filtered_svs if s['type'] == sv_type and s['true_positive'])
    type_counts[sv_type] = n
    print(f"  {sv_type}: {n} ({100*n/max(tp_filtered,1):.1f}%)")

# Size distribution
sizes = [s['size'] for s in filtered_svs if s['true_positive'] and s['type'] != 'TRA']
print(f"\n  SV size statistics:")
print(f"  Median: {np.median(sizes)/1000:.1f} kb")
print(f"  Range: {min(sizes)/1000:.1f} - {max(sizes)/1000:.1f} kb")

# ─── 6. GENOTYPING ───────────────────────────────────────────
print("\n[Genotyping] SV genotyping...")

# Genotype from VAF
def genotype_sv(vaf, depth=30):
    """Genotype SV based on VAF."""
    # Binomial likelihood for HET (VAF~0.5) vs HOM (VAF~1.0)
    n_alt = int(vaf * depth)
    p_het = stats.binom.pmf(n_alt, depth, 0.5)
    p_hom = stats.binom.pmf(n_alt, depth, 0.95)
    if p_hom > p_het:
        return 'HOM', p_hom / (p_het + p_hom)
    else:
        return 'HET', p_het / (p_het + p_hom)

genotype_results = [genotype_sv(s['vaf']) for s in filtered_svs if s['true_positive']]
n_het = sum(1 for g, _ in genotype_results if g == 'HET')
n_hom = sum(1 for g, _ in genotype_results if g == 'HOM')
print(f"  HET: {n_het} ({100*n_het/max(len(genotype_results),1):.1f}%)")
print(f"  HOM: {n_hom} ({100*n_hom/max(len(genotype_results),1):.1f}%)")

# ─── 7. CANCER DRIVER SV ANALYSIS ────────────────────────────
print("\n[Cancer] Identifying cancer driver SVs...")

# Known cancer driver genes (simplified)
CANCER_GENES = {
    'TP53': (17, 7_668_421, 7_687_490),
    'BRCA1': (17, 43_044_295, 43_125_483),
    'BRCA2': (13, 32_315_086, 32_400_268),
    'MYC': (8, 127_735_434, 127_742_951),
    'EGFR': (7, 55_019_017, 55_211_628),
    'PTEN': (10, 89_623_195, 89_728_532),
    'RB1': (13, 48_303_747, 49_060_518),
    'CDKN2A': (9, 21_967_752, 21_995_324),
    'BCR': (22, 23_179_704, 23_318_037),
    'ABL1': (9, 130_713_016, 130_887_675),
}

def overlaps_gene(sv, gene_chrom, gene_start, gene_end):
    """Check if SV breakpoint overlaps a gene."""
    if sv['chrom1'] == gene_chrom:
        if gene_start <= sv['pos1'] <= gene_end:
            return True
    if sv['chrom2'] == gene_chrom:
        if gene_start <= sv['pos2'] <= gene_end:
            return True
    return False

driver_svs = []
for sv in filtered_svs:
    if not sv['true_positive']:
        continue
    for gene, (chrom, start, end) in CANCER_GENES.items():
        if overlaps_gene(sv, chrom, start, end):
            driver_svs.append({**sv, 'driver_gene': gene})
            break

print(f"  SVs disrupting cancer driver genes: {len(driver_svs)}")
gene_disruption_counts = {}
for dsv in driver_svs:
    g = dsv['driver_gene']
    gene_disruption_counts[g] = gene_disruption_counts.get(g, 0) + 1

for gene, count in sorted(gene_disruption_counts.items(), key=lambda x: -x[1]):
    print(f"  {gene}: {count} SVs")

# BCR-ABL1 translocation detection
bcr_abl_candidates = [s for s in filtered_svs
                      if s['type'] == 'TRA' and
                      ((s['chrom1'] == 22 and s['chrom2'] == 9) or
                       (s['chrom1'] == 9 and s['chrom2'] == 22))]
print(f"\n  BCR-ABL1 translocation candidates: {len(bcr_abl_candidates)}")

# ─── 8. VISUALIZATION ────────────────────────────────────────
print("\n[Viz] Generating dashboard...")

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0a0a0a')
gs_main = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

sv_colors = {'DEL': '#FF5722', 'DUP': '#2196F3', 'INV': '#4CAF50',
             'TRA': '#9C27B0', 'INS': '#FF9800'}

# Panel 1: Circos-style chromosome plot
ax1 = fig.add_subplot(gs_main[0, 0], projection='polar')
ax1.set_facecolor('#111111')
theta = np.linspace(0, 2*np.pi, N_CHROMOSOMES + 1)[:-1]
chrom_arc = CHROM_SIZES / CHROM_SIZES.sum() * 2 * np.pi * 0.9
for i, (t, arc) in enumerate(zip(theta, chrom_arc)):
    ax1.bar(t, 1, width=arc * 0.9, bottom=0.5, color=f'#{hash(i*7)%0xFFFFFF:06X}', alpha=0.6)
# Plot TRA SVs as arcs
for sv in filtered_svs[:20]:
    if sv['type'] == 'TRA' and sv['true_positive']:
        t1 = theta[sv['chrom1'] - 1]
        t2 = theta[sv['chrom2'] - 1]
        ax1.plot([t1, t2], [1.2, 1.2], color='#E9ED4C', alpha=0.4, linewidth=0.5)
ax1.set_title('Chromosome SV Map', color='white', fontsize=10, fontweight='bold', pad=15)
ax1.tick_params(colors='white', labelsize=6)
ax1.set_yticklabels([])

# Panel 2: SV type distribution
ax2 = fig.add_subplot(gs_main[0, 1])
ax2.set_facecolor('#111111')
bars = ax2.bar(type_counts.keys(), type_counts.values(),
               color=[sv_colors[t] for t in type_counts.keys()], alpha=0.85)
ax2.set_ylabel('Count', color='white', fontsize=9)
ax2.set_title('SV Type Distribution', color='white', fontsize=10, fontweight='bold')
ax2.tick_params(colors='white', labelsize=8)
for spine in ax2.spines.values():
    spine.set_color('#333333')
for bar in bars:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(int(bar.get_height())), ha='center', va='bottom', color='white', fontsize=8)

# Panel 3: SV size distribution
ax3 = fig.add_subplot(gs_main[0, 2])
ax3.set_facecolor('#111111')
for sv_type in ['DEL', 'DUP', 'INV']:
    type_sizes = [s['size'] for s in filtered_svs if s['type'] == sv_type and s['true_positive'] and s['size'] > 0]
    if type_sizes:
        ax3.hist(np.log10(type_sizes), bins=20, alpha=0.6, color=sv_colors[sv_type], label=sv_type)
ax3.set_xlabel('log10(SV size, bp)', color='white', fontsize=9)
ax3.set_ylabel('Count', color='white', fontsize=9)
ax3.set_title('SV Size Distribution', color='white', fontsize=10, fontweight='bold')
ax3.tick_params(colors='white', labelsize=7)
for spine in ax3.spines.values():
    spine.set_color('#333333')
ax3.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 4: Precision-Recall
ax4 = fig.add_subplot(gs_main[1, 0])
ax4.set_facecolor('#111111')
# Vary quality threshold
thresholds = range(5, 55, 5)
precisions, recalls, f1s = [], [], []
for qt in thresholds:
    filt = [s for s in detected_svs if s['quality'] >= qt]
    tp = sum(1 for s in filt if s['true_positive'])
    fp = sum(1 for s in filt if not s['true_positive'])
    fn_t = len(all_true_svs) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn_t) if (tp + fn_t) > 0 else 0
    f = 2*p*r/(p+r) if (p+r) > 0 else 0
    precisions.append(p)
    recalls.append(r)
    f1s.append(f)
ax4.plot(recalls, precisions, 'o-', color='#FF5722', linewidth=2, markersize=5)
ax4.set_xlabel('Recall', color='white', fontsize=9)
ax4.set_ylabel('Precision', color='white', fontsize=9)
ax4.set_title(f'Precision-Recall (F1={f1:.3f})', color='white', fontsize=10, fontweight='bold')
ax4.set_xlim(0, 1.05)
ax4.set_ylim(0, 1.05)
ax4.tick_params(colors='white', labelsize=7)
for spine in ax4.spines.values():
    spine.set_color('#333333')

# Panel 5: VAF distribution
ax5 = fig.add_subplot(gs_main[1, 1])
ax5.set_facecolor('#111111')
vafs_tp = [s['vaf'] for s in filtered_svs if s['true_positive']]
vafs_fp = [s['vaf'] for s in filtered_svs if not s['true_positive']]
ax5.hist(vafs_tp, bins=20, color='#4CAF50', alpha=0.7, label=f'TP (n={len(vafs_tp)})')
ax5.hist(vafs_fp, bins=10, color='#FF5722', alpha=0.7, label=f'FP (n={len(vafs_fp)})')
ax5.axvline(x=0.5, color='yellow', linestyle='--', linewidth=0.8, label='VAF=0.5')
ax5.set_xlabel('Variant Allele Frequency', color='white', fontsize=9)
ax5.set_ylabel('Count', color='white', fontsize=9)
ax5.set_title('VAF Distribution', color='white', fontsize=10, fontweight='bold')
ax5.tick_params(colors='white', labelsize=7)
for spine in ax5.spines.values():
    spine.set_color('#333333')
ax5.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 6: Cancer driver genes
ax6 = fig.add_subplot(gs_main[1, 2])
ax6.set_facecolor('#111111')
if gene_disruption_counts:
    genes_sorted = sorted(gene_disruption_counts.items(), key=lambda x: -x[1])
    g_names = [g for g, _ in genes_sorted]
    g_counts = [c for _, c in genes_sorted]
    bars6 = ax6.barh(g_names, g_counts, color='#E91E63', alpha=0.85)
    ax6.set_xlabel('SV Count', color='white', fontsize=9)
    ax6.set_title('Cancer Driver Gene Disruptions', color='white', fontsize=10, fontweight='bold')
    ax6.tick_params(colors='white', labelsize=8)
    for spine in ax6.spines.values():
        spine.set_color('#333333')

# Panel 7: Quality score distribution
ax7 = fig.add_subplot(gs_main[2, 0])
ax7.set_facecolor('#111111')
q_tp = [s['quality'] for s in detected_svs if s['true_positive']]
q_fp = [s['quality'] for s in detected_svs if not s['true_positive']]
ax7.hist(q_tp, bins=20, color='#4CAF50', alpha=0.7, label='True SVs')
ax7.hist(q_fp, bins=15, color='#FF5722', alpha=0.7, label='False SVs')
ax7.axvline(x=quality_threshold, color='yellow', linestyle='--', linewidth=1, label=f'Threshold={quality_threshold}')
ax7.set_xlabel('Quality Score', color='white', fontsize=9)
ax7.set_ylabel('Count', color='white', fontsize=9)
ax7.set_title('Quality Score Distribution', color='white', fontsize=10, fontweight='bold')
ax7.tick_params(colors='white', labelsize=7)
for spine in ax7.spines.values():
    spine.set_color('#333333')
ax7.legend(fontsize=7, facecolor='#222222', labelcolor='white')

# Panel 8: Genotype distribution
ax8 = fig.add_subplot(gs_main[2, 1])
ax8.set_facecolor('#111111')
ax8.pie([n_het, n_hom], labels=['HET', 'HOM'], colors=['#2196F3', '#FF5722'],
        autopct='%1.0f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 10})
ax8.set_title('SV Genotype Distribution', color='white', fontsize=10, fontweight='bold')

# Panel 9: Summary
ax9 = fig.add_subplot(gs_main[2, 2])
ax9.set_facecolor('#111111')
ax9.axis('off')
summary = [
    "StructuralVariantEngine v1.0",
    "",
    f"Samples: {N_SAMPLES} tumor-normal",
    f"True SVs: {len(all_true_svs)}",
    f"Detected: {len(detected_svs)}",
    f"After QC: {len(filtered_svs)}",
    "",
    f"Performance:",
    f"  Precision: {precision:.3f}",
    f"  Recall: {recall:.3f}",
    f"  F1: {f1:.3f}",
    "",
    f"SV types: DEL={type_counts.get('DEL',0)}, DUP={type_counts.get('DUP',0)}",
    f"  INV={type_counts.get('INV',0)}, TRA={type_counts.get('TRA',0)}",
    "",
    f"Driver gene SVs: {len(driver_svs)}",
    f"BCR-ABL1 candidates: {len(bcr_abl_candidates)}",
]
for i, line in enumerate(summary):
    color = '#E9ED4C' if i == 0 else 'white'
    ax9.text(0.05, 0.97 - i * 0.055, line, transform=ax9.transAxes,
             color=color, fontsize=8.5, va='top',
             fontweight='bold' if i == 0 else 'normal')

fig.suptitle('StructuralVariantEngine: SV Detection and Analysis Dashboard',
             color='white', fontsize=14, fontweight='bold', y=0.98)

plt.savefig('/workspace/sv_dashboard.png', dpi=150, bbox_inches='tight',
            facecolor='#0a0a0a')
plt.close()
print("  Dashboard saved.")

print("\n" + "=" * 60)
print("StructuralVariantEngine COMPLETE")
print(f"  Samples: {N_SAMPLES} | True SVs: {len(all_true_svs)}")
print(f"  Detected: {len(detected_svs)} → Filtered: {len(filtered_svs)}")
print(f"  Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
print(f"  Driver gene SVs: {len(driver_svs)}")
print(f"  BCR-ABL1 candidates: {len(bcr_abl_candidates)}")
print("=" * 60)
