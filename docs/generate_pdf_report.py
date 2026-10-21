import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.pdfgen import canvas

# ----------------------------------------------------------------------
# Numbered Canvas for dynamic page counts (Page X of Y)
# ----------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # First page: simple top accent bar
            self.saveState()
            self.setFillColor(colors.HexColor("#1E3A8A"))
            self.rect(54, 11 * 72 - 20, 8.5 * 72 - 108, 6, fill=True, stroke=False)
            self.restoreState()
            return

        self.saveState()
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Header
        self.drawString(54, 11 * 72 - 36, "ST-GoG PROJECT: TECHNICAL AUDIT & ONBOARDING REPORT")
        self.setFont("Helvetica", 8)
        self.drawRightString(8.5 * 72 - 54, 11 * 72 - 36, "INTERNAL RESEARCH & ONBOARDING REPORT")
        
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(54, 11 * 72 - 42, 8.5 * 72 - 54, 11 * 72 - 42)

        # Running Footer
        self.line(54, 40, 8.5 * 72 - 54, 40)
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        self.drawString(54, 28, "MTSC Deep Learning Research — ABV-IIITM Gwalior")
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * 72 - 54, 28, page_text)
        self.restoreState()


def build_pdf(filename="d:/first intern(maybe)/ST_GoG_Project_Analysis_and_Onboarding_Report.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    primary_color = colors.HexColor("#1E3A8A")   # Deep Navy
    secondary_color = colors.HexColor("#0D9488") # Teal
    dark_neutral = colors.HexColor("#0F172A")    # Slate 900
    body_color = colors.HexColor("#334155")      # Slate 700
    border_color = colors.HexColor("#E2E8F0")

    styles.add(ParagraphStyle(
        name='CoverPreTitle',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13,
        textColor=secondary_color,
        spaceAfter=4
    ))

    styles.add(ParagraphStyle(
        name='DocTitle',
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name='DocSubtitle',
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=body_color,
        spaceAfter=10
    ))

    styles.add(ParagraphStyle(
        name='SectionHeader',
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='SubSectionHeader',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13.5,
        textColor=secondary_color,
        spaceBefore=7,
        spaceAfter=3,
        keepWithNext=True
    ))

    styles.add(ParagraphStyle(
        name='BodyCustom',
        fontName='Helvetica',
        fontSize=8.6,
        leading=12.0,
        textColor=body_color,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name='BulletCustom',
        fontName='Helvetica',
        fontSize=8.4,
        leading=11.6,
        textColor=body_color,
        leftIndent=12,
        spaceAfter=2.5
    ))

    styles.add(ParagraphStyle(
        name='CalloutText',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.8,
        textColor=colors.HexColor("#1E293B")
    ))

    styles.add(ParagraphStyle(
        name='TableHeader',
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    ))

    styles.add(ParagraphStyle(
        name='TableCell',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=dark_neutral,
        alignment=0
    ))

    styles.add(ParagraphStyle(
        name='TableCellCenter',
        fontName='Helvetica',
        fontSize=7.5,
        leading=9.5,
        textColor=dark_neutral,
        alignment=1
    ))

    styles.add(ParagraphStyle(
        name='TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=primary_color,
        alignment=1
    ))

    story = []

    # =========================================================================
    # PAGE 1: TITLE, METADATA, EXECUTIVE SUMMARY & PROBLEM CONTEXT
    # =========================================================================
    story.append(Spacer(1, 8))
    story.append(Paragraph("PROJECT TECHNICAL AUDIT & ONBOARDING GUIDE", styles['CoverPreTitle']))
    story.append(Paragraph("Spatiotemporal Graph-of-Graphs (ST-GoG) for Multivariate Time Series Classification", styles['DocTitle']))
    story.append(Paragraph("Comprehensive Analysis of Theory, Empirical Benchmarks, Codebase Health, and Strategic Action Plan", styles['DocSubtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceBefore=0, spaceAfter=8))

    meta_data = [
        [
            Paragraph("<b>Target Audience:</b> Incoming Engineering & Research Intern", styles['TableCell']),
            Paragraph("<b>Origin:</b> M.Tech Thesis Handover (Jan 2026 – May 2026)", styles['TableCell'])
        ],
        [
            Paragraph("<b>Primary Author:</b> Tushar Sharma (ABV-IIITM Gwalior)", styles['TableCell']),
            Paragraph("<b>Supervisor:</b> Dr. Anjali (Assistant Professor, Dept of IT)", styles['TableCell'])
        ],
        [
            Paragraph("<b>Framework:</b> PyTorch 2.1, CUDA AMP, sktime", styles['TableCell']),
            Paragraph("<b>Evaluation Scope:</b> 22 UEA Multivariate Datasets", styles['TableCell'])
        ]
    ]
    meta_table = Table(meta_data, colWidths=[255, 261])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F8FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#CBD5E1")),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    exec_text = """<b>EXECUTIVE SUMMARY & ONBOARDING CONTEXT:</b><br/>
You have been assigned to inherit and advance an ongoing deep learning research project aimed at <b>Multivariate Time Series Classification (MTSC)</b>. The project is detailed in a completed 55-page M.Tech thesis and substantiated by experimental notebooks. The proposed <b>ST-GoG</b> framework unites dilated residual temporal convolutions, multi-scale graph convolutional networks, and hierarchical segment-level graph aggregation into an end-to-end differentiable pipeline. Across 22 benchmark datasets, ST-GoG outperforms published SOTA baselines (SAGoG, ROCKET) in mean accuracy (74.3%) and total wins (10). However, the existing code is composed of uncurated Kaggle notebook cells with hardcoded environment paths, unhandled long-sequence bottlenecks, and fragile multi-GPU code. This briefing provides everything you need to hit the ground running with 100% technical clarity."""
    
    callout_table = Table([[Paragraph(exec_text, styles['CalloutText'])]], colWidths=[516])
    callout_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#EFF6FF")),
        ('BOX', (0,0), (-1,-1), 1.2, colors.HexColor("#3B82F6")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(callout_table)
    story.append(Spacer(1, 8))

    story.append(Paragraph("1. Fundamental Problem & Research Motivation", styles['SectionHeader']))
    story.append(Paragraph(
        "<b>What is Multivariate Time Series Classification (MTSC)?</b><br/>"
        "In MTSC, an input sample is a synchronized matrix <b>X &isin; R^(C &times; T)</b> recorded from <b>C</b> sensors (e.g. EEG electrodes, IMU sensors, traffic detectors) over <b>T</b> discrete time steps. The task is to assign a discrete class label <i>y</i> to the entire recording. Applications span medical diagnosis (cardiac arrhythmia, seizure onset), aviation gesture recognition, automated handwriting verification, and industrial equipment telemetry.",
        styles['BodyCustom']
    ))
    story.append(Paragraph(
        "<b>The Core Technical Bottleneck: Tri-Fold Dependencies</b><br/>"
        "Time-series models typically excel at either temporal filtering or spatial graph topology, but rarely both. Real signals exhibit three distinct dependency structures that must be resolved simultaneously:<br/>"
        "&bull; <b>Intra-Channel Temporal Dynamics:</b> Long-range temporal phase alignments and transient local wavelets.<br/>"
        "&bull; <b>Inter-Channel Spatial Relational Dynamics:</b> Meaningful patterns arise from cross-sensor interactions rather than any channel in isolation.<br/>"
        "&bull; <b>Temporal Non-Stationarity:</b> Inter-channel coupling changes as recordings evolve (e.g. gesture motion phases).",
        styles['BodyCustom']
    ))
    story.append(Paragraph(
        "<b>Limitations of Prior Art:</b> <i>ROCKET</i> extracts 10k random convolutional features with ridge regression; while fast on periodic data, it treats channels independently without structural relational reasoning. <i>SAGoG</i> constructs a complete pairwise instance similarity graph using Dynamic Time Warping (DTW), yielding crippling <i>O(N^2)</i> memory complexity that bottlenecks large cohorts. ST-GoG solves both trade-offs.",
        styles['BodyCustom']
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: THE PROPOSED ST-GoG ARCHITECTURE
    # =========================================================================
    story.append(Paragraph("2. The Proposed ST-GoG Framework Architecture", styles['SectionHeader']))
    story.append(Paragraph(
        "ST-GoG decouples feature extraction into parallel temporal and spatial representation backbones, feeds both into hierarchical segment-level graph modules, and merges them through learned softmax pathway fusion.",
        styles['BodyCustom']
    ))

    if os.path.exists('d:/first intern(maybe)/fig_architecture.png'):
        story.append(Image('d:/first intern(maybe)/fig_architecture.png', width=510, height=160))
        story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Architectural Components & Mathematical Formulations:</b>", styles['SubSectionHeader']))

    arch_points = [
        "<b>1. Residual TCN with Squeeze-and-Excitation (ResTCN+SE):</b> Five residual blocks with dilations <i>d &isin; {1, 2, 4, 8, 16}</i> and kernel size <i>K=3</i> provide an effective receptive field of <b>63 time steps</b> without subsampling. A channel-wise SE attention mechanism adaptively rescales feature maps via global average pooling followed by a 2-layer bottleneck MLP: <i>w_se = &sigma;(W_2 &delta;(W_1 AvgPool(h)))</i>, suppressing sensor noise.",
        "<b>2. Multi-Scale GCN with Learnable Softmax Fusion:</b> Rather than computing dense <i>O(C^2)</i> feature similarities, ST-GoG builds symmetric normalized chain adjacency matrices <i>A_r</i> at three distinct temporal radii: local (<i>r=1</i>), medium (<i>r=3</i>), and global (<i>r=5</i>). Three parallel 3-layer GCN branches extract spatial motifs and are merged via learned softmax weights: <i>H_multi = &sum; Softmax(w)_i GCN(H^(0), A_ri)</i>.",
        "<b>3. Hierarchical Graph-of-Graphs (GoG) Module:</b> To model evolving temporal phases, the time sequence is split into <i>S</i> uniform temporal segments (adaptively selected: <i>S=4</i> for <i>T &lt; 100</i>; <i>S=6</i> for <i>T &lt; 200</i>; <i>S=8</i> for <i>T &lt; 400</i>; <i>S=10</i> for <i>T &ge; 400</i>). An <b>Inner GCN (radius r=3)</b> operates within each segment sub-graph, followed by mean pooling to form segment super-nodes. An <b>Outer GCN (radius r=1)</b> then connects consecutive super-nodes to capture macro-level transitions across time.",
        "<b>4. Four-Pathway Softmax Fusion Head:</b> To prevent information loss, four representation streams are preserved: (1) TCN global pool, (2) TCN + GoG, (3) GCN global pool, and (4) GCN + GoG. Learned softmax coefficients <i>&alpha; = [&alpha;1, &alpha;2, &alpha;3, &alpha;4]</i> blend these vectors before passing to a 2-block MLP classifier with batch normalization and dropout (0.42).",
        "<b>5. Regularization & Optimization Engine:</b> Employs MixUp data augmentation (<i>&alpha; = 0.2</i>), label smoothing (0.10 &ndash; 0.15), Cosine Annealing with Warm Restarts (<i>T_0=50, T_mult=2</i>), exponential stochastic depth decay, snapshot ensembling across restart minima, and 7-pass Test-Time Augmentation (TTA)."
    ]
    for ap in arch_points:
        story.append(Paragraph(f"&bull; {ap}", styles['BulletCustom']))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: BENCHMARK RESULTS ACROSS 22 DATASETS
    # =========================================================================
    story.append(Paragraph("3. Empirical Benchmark Results & Comparative Analysis", styles['SectionHeader']))
    story.append(Paragraph(
        "ST-GoG was benchmarked against SAGoG, ROCKET, and the Simple Parallel baseline across <b>22 UEA datasets</b> covering Motion, Medical, High-Dimensional Sensors, and Astronomy.",
        styles['BodyCustom']
    ))

    table_data = [
        [
            Paragraph("<b>Dataset</b>", styles['TableHeader']),
            Paragraph("<b>Domain</b>", styles['TableHeader']),
            Paragraph("<b>Train (N)</b>", styles['TableHeader']),
            Paragraph("<b>Ch (C)</b>", styles['TableHeader']),
            Paragraph("<b>Len (T)</b>", styles['TableHeader']),
            Paragraph("<b>SAGoG</b>", styles['TableHeader']),
            Paragraph("<b>ROCKET</b>", styles['TableHeader']),
            Paragraph("<b>Simple Par.</b>", styles['TableHeader']),
            Paragraph("<b>ST-GoG (Ours)</b>", styles['TableHeader']),
        ]
    ]

    benchmarks = [
        ("ArticularyWordRec", "Motion", "275", "9", "144", "94.6%", "99.6%", "98.0%", "<b>99.7%*</b>"),
        ("AtrialFibrillation", "Medical", "15", "2", "640", "43.0%", "24.9%", "40.0%", "<b>53.3%*</b>"),
        ("DuckDuckGeese", "Audio", "50", "1345", "270", "<b>79.0%*</b>", "46.1%", "74.0%", "72.0%"),
        ("EigenWorms", "Biology", "128", "6", "17984", "81.5%", "<b>86.3%*</b>", "53.4%", "59.5%"),
        ("Epilepsy", "Medical", "137", "3", "206", "92.8%", "<b>99.1%*</b>", "53.4%", "98.6%"),
        ("FaceDetection", "EEG", "5890", "144", "62", "68.3%", "<b>69.4%*</b>", "67.6%", "53.0%"),
        ("FingerMovements", "EEG", "316", "28", "50", "61.0%", "55.3%", "58.0%", "<b>64.0%*</b>"),
        ("Handwriting", "Motion", "150", "3", "152", "64.3%", "56.7%", "28.7%", "<b>66.0%*</b>"),
        ("Heartbeat", "Medical", "204", "61", "405", "76.1%", "71.8%", "79.0%", "<b>79.5%*</b>"),
        ("Libras", "Motion", "180", "2", "45", "87.1%", "90.6%", "91.1%", "<b>95.6%*</b>"),
        ("LSST", "Astro", "2459", "6", "36", "<b>66.9%*</b>", "63.2%", "64.0%", "65.7%"),
        ("NATOPS", "Aviation", "180", "24", "51", "97.2%", "88.5%", "97.8%", "<b>99.4%*</b>"),
        ("PenDigits", "Motion", "7494", "2", "8", "94.6%", "<b>99.6%*</b>", "99.0%", "99.2%"),
        ("PEMS-SF", "MISC", "267", "963", "144", "81.2%", "85.6%", "87.9%", "<b>87.9%*</b>"),
        ("PhonemeSpectra", "Motion", "3000", "54", "1000", "<b>34.1%*</b>", "28.4%", "27.8%", "31.2%"),
        ("SelfRegulationSCP1", "EEG", "268", "6", "896", "91.0%", "86.6%", "88.4%", "<b>91.8%*</b>"),
        ("SelfRegulationSCP2", "EEG", "200", "7", "1152", "58.9%", "51.4%", "58.3%", "<b>59.4%*</b>"),
        ("StandWalkJump", "Medical", "12", "4", "2500", "<b>64.8%*</b>", "45.6%", "53.3%", "60.0%"),
        ("UWaveGestureLibrary", "Motion", "120", "3", "315", "85.9%", "<b>94.4%*</b>", "78.4%", "92.2%"),
        ("OVERALL AVERAGE", "22 Bench", "---", "---", "---", "74.0%", "65.7%", "69.7%", "<b>74.3%*</b>"),
    ]

    for row in benchmarks:
        is_avg = row[0] == "OVERALL AVERAGE"
        bg_style = styles['TableCellBold'] if is_avg else styles['TableCell']
        center_style = styles['TableCellBold'] if is_avg else styles['TableCellCenter']
        table_data.append([
            Paragraph(f"<b>{row[0]}</b>" if is_avg else row[0], bg_style),
            Paragraph(row[1], center_style),
            Paragraph(row[2], center_style),
            Paragraph(row[3], center_style),
            Paragraph(row[4], center_style),
            Paragraph(row[5], center_style),
            Paragraph(row[6], center_style),
            Paragraph(row[7], center_style),
            Paragraph(row[8], center_style),
        ])

    b_table = Table(table_data, colWidths=[112, 54, 42, 34, 40, 50, 52, 56, 76])
    b_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('TOPPADDING', (0,0), (-1,-1), 2.2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.2),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor("#F8FAFC")]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor("#E0E7FF")),
        ('LINEABOVE', (0,-1), (-1,-1), 1.2, primary_color),
    ]))
    story.append(b_table)
    story.append(Spacer(1, 6))

    if os.path.exists('d:/first intern(maybe)/fig_benchmark_comparison.png'):
        story.append(Image('d:/first intern(maybe)/fig_benchmark_comparison.png', width=510, height=200))
        story.append(Spacer(1, 4))

    story.append(Paragraph(
        "<b>Performance Insights:</b> ST-GoG earns the best average rank of <b>1.73</b> (Wilcoxon signed-rank test confirmed statistical parity with SAGoG but significant superiority over ROCKET and Simple Parallel). It wins outright on <b>10 out of 22 datasets</b>, showing exceptional strength on small sample sizes (e.g. <i>AtrialFibrillation</i> with N=15, beating SAGoG by +10.33%) and structured kinematics (<i>NATOPS</i> 99.4%, <i>Libras</i> 95.6%).",
        styles['BodyCustom']
    ))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: ABLATION STUDY & SYSTEMATIC FAILURE ANALYSIS
    # =========================================================================
    story.append(Paragraph("4. Ablation Study & Systematic Failure Analysis", styles['SectionHeader']))
    story.append(Paragraph(
        "The ablation study on <b>PEMS-SF</b> (C=963, T=144, N=267) isolates the exact accuracy contribution and parameter efficiency of each architectural module.",
        styles['BodyCustom']
    ))

    if os.path.exists('d:/first intern(maybe)/fig_ablation_breakdown.png'):
        story.append(Image('d:/first intern(maybe)/fig_ablation_breakdown.png', width=490, height=190))
        story.append(Spacer(1, 6))

    ablation_data = [
        [
            Paragraph("<b>Architecture Variant</b>", styles['TableHeader']),
            Paragraph("<b>Target Component Removed</b>", styles['TableHeader']),
            Paragraph("<b>Parameters</b>", styles['TableHeader']),
            Paragraph("<b>Best Acc. (%)</b>", styles['TableHeader']),
            Paragraph("<b>&Delta; vs Full</b>", styles['TableHeader']),
        ],
        [Paragraph("<b>Full ST-GoG (V1)</b>", styles['TableCellBold']), Paragraph("None (Complete Model)", styles['TableCell']), Paragraph("1,908,041", styles['TableCellCenter']), Paragraph("87.28%", styles['TableCellCenter']), Paragraph("Baseline", styles['TableCellCenter'])],
        [Paragraph("w/o Attention Pool (V2)", styles['TableCell']), Paragraph("Robust Attention Pooling &rarr; Mean Pooling", styles['TableCell']), Paragraph("1,883,075", styles['TableCellCenter']), Paragraph("87.86%", styles['TableCellCenter']), Paragraph("+0.58%", styles['TableCellCenter'])],
        [Paragraph("w/o SE Attention (V3)", styles['TableCell']), Paragraph("Squeeze-Excitation blocks stripped", styles['TableCell']), Paragraph("1,899,689", styles['TableCellCenter']), Paragraph("86.71%", styles['TableCellCenter']), Paragraph("-0.57%", styles['TableCellCenter'])],
        [Paragraph("w/o Multi-Scale GCN (V4)", styles['TableCell']), Paragraph("Replaced by single radius GCN (r=3)", styles['TableCell']), Paragraph("1,561,408", styles['TableCellCenter']), Paragraph("86.71%", styles['TableCellCenter']), Paragraph("-0.57%", styles['TableCellCenter'])],
        [Paragraph("w/o GoG Module (V5)", styles['TableCell']), Paragraph("Stripped Graph-of-Graphs (Global Pool only)", styles['TableCell']), Paragraph("1,494,583", styles['TableCellCenter']), Paragraph("<b>84.39%</b>", styles['TableCellBold']), Paragraph("<b>-2.89% &darr;</b>", styles['TableCellBold'])],
    ]

    ab_table = Table(ablation_data, colWidths=[130, 176, 70, 70, 70])
    ab_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(ab_table)
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Where Does ST-GoG Fail? (Three Critical Bottlenecks to Fix):</b>", styles['SubSectionHeader']))
    fail_points = [
        "<b>1. Extreme Sequence Length (EigenWorms, T=17,984):</b> ST-GoG achieved 59.49% vs ROCKET's 86.28%. Splitting 17,984 time steps into 10 segments results in inner GCN sub-graphs with ~1,800 nodes each. Graph convolution on 1,800 nodes causes massive VRAM consumption and slow epoch iterations, preventing convergence within the compute budget.",
        "<b>2. Dense Cohort Bottleneck (FaceDetection, N=5,890, C=144):</b> ST-GoG achieved 53.04% vs ROCKET's 69.42% and SAGoG's 68.30%. Constructing graph matrices for nearly 6,000 instances creates an epoch execution bottleneck. Furthermore, EEG electrodes exhibit complex circular spatial layouts that chain adjacency cannot capture.",
        "<b>3. High Channel Counts with Data Scarcity (DuckDuckGeese, C=1,345, N=50):</b> SAGoG achieved 79.00% vs ST-GoG's 72.00%. In audio spectrograms with >1,300 channels, harmonic relationships are non-local; chain adjacency fails to connect distant resonant frequency bands."
    ]
    for fp in fail_points:
        story.append(Paragraph(f"&bull; {fp}", styles['BulletCustom']))

    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: CODEBASE AUDIT & TECHNICAL DEBT
    # =========================================================================
    story.append(Paragraph("5. Existing Codebase Audit & Technical Debt", styles['SectionHeader']))
    story.append(Paragraph(
        "The code in <code>stgogcompleteworkfromscratch/</code> consists of 7 raw Jupyter notebook exports. Below is an objective audit of their operational viability:",
        styles['BodyCustom']
    ))

    code_audit = [
        [
            Paragraph("<b>File on Disk</b>", styles['TableHeader']),
            Paragraph("<b>Size</b>", styles['TableHeader']),
            Paragraph("<b>Contents & Implementation Details</b>", styles['TableHeader']),
            Paragraph("<b>Operational Status</b>", styles['TableHeader'])
        ],
        [
            Paragraph("<code>stgog.ipynb.txt</code>", styles['TableCellBold']),
            Paragraph("118 KB", styles['TableCellCenter']),
            Paragraph("Main research notebook: single-dataset run + 22-dataset batch loop on Kaggle 2x T4.", styles['TableCell']),
            Paragraph("<font color='#0D9488'><b>Working:</b></font> Fully functional for Kaggle multi-GPU runs. Contains full model & snapshot logic.", styles['TableCell'])
        ],
        [
            Paragraph("<code>stgog.ipynb (1).txt</code>", styles['TableCell']),
            Paragraph("156 KB", styles['TableCellCenter']),
            Paragraph("Duplicate copy of stgog.ipynb.txt with saved execution outputs.", styles['TableCell']),
            Paragraph("<font color='#64748B'><b>Redundant:</b></font> Duplicate artifact.", styles['TableCell'])
        ],
        [
            Paragraph("<code>notebook19a6f5ec01.ipynb.txt</code>", styles['TableCell']),
            Paragraph("118 KB", styles['TableCellCenter']),
            Paragraph("Exact clone of stgog.ipynb.txt exported from Kaggle kernel.", styles['TableCell']),
            Paragraph("<font color='#64748B'><b>Redundant:</b></font> Duplicate artifact.", styles['TableCell'])
        ],
        [
            Paragraph("<code>notebookb57bcbeca7.ipynb.txt</code>", styles['TableCellBold']),
            Paragraph("43 KB", styles['TableCellCenter']),
            Paragraph("Ablation study script evaluating V1..V6 variants on PEMS-SF.", styles['TableCell']),
            Paragraph("<font color='#0D9488'><b>Working:</b></font> Clean code, reproduced Table 5.9 ablation metrics.", styles['TableCell'])
        ],
        [
            Paragraph("<code>Untitled3.ipynb.txt</code>", styles['TableCell']),
            Paragraph("348 KB", styles['TableCellCenter']),
            Paragraph("Early baseline prototype using torch_geometric on PenDigits (88.5% acc).", styles['TableCell']),
            Paragraph("<font color='#EAB308'><b>Legacy:</b></font> Uses old PyG dependencies abandoned in favor of pure PyTorch.", styles['TableCell'])
        ],
        [
            Paragraph("<code>Untitled5.ipynb.txt</code>", styles['TableCellBold']),
            Paragraph("615 KB", styles['TableCellCenter']),
            Paragraph("Extensive experimental trials on FaceDetection, StandWalkJump, PhonemeSpectra, LSST.", styles['TableCell']),
            Paragraph("<font color='#0D9488'><b>Valuable:</b></font> Contains debugged learning rate schedules and batch loss curves.", styles['TableCell'])
        ],
        [
            Paragraph("<code>Untitled6.ipynb.txt</code>", styles['TableCell']),
            Paragraph("9.5 KB", styles['TableCellCenter']),
            Paragraph("Experimental draft titled 'FINAL ROCKET-BEATING UNIVERSAL MTSC MODEL'.", styles['TableCell']),
            Paragraph("<font color='#DC2626'><b>Broken:</b></font> Calls undefined function <code>topk_adj()</code>; execution fails with NameError.", styles['TableCell'])
        ]
    ]

    c_table = Table(code_audit, colWidths=[124, 40, 202, 150])
    c_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), primary_color),
        ('GRID', (0,0), (-1,-1), 0.5, border_color),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F8FAFC")]),
    ]))
    story.append(c_table)
    story.append(Spacer(1, 8))

    code_warn = """<b>TECHNICAL DEBT & DISCREPANCIES DISCOVERED:</b><br/>
1. <b>Zero Modular Architecture:</b> There is no Python package structure (no <code>setup.py</code>, <code>src/</code>, <code>models/</code>, or CLI entry point). Everything exists as 500-line monolithic notebook cells.<br/>
2. <b>Environment & Platform Lock-In:</b> Notebooks contain hardcoded paths (e.g. <code>/kaggle/working</code>), shell commands (<code>!pip install</code>), and assume dual-GPU <code>torch.nn.DataParallel</code>. Running this on a local machine or single-GPU workstation causes batch chunk errors.<br/>
3. <b>Discrepancy in Thesis Documentation:</b> In Table 5.9 of the report, Variant 2 is labeled <i>'w/o Training Aug. Pipeline'</i>, but in the actual codebase (<code>notebookb57bcbeca7.ipynb.txt</code>), Variant 2 is actually <i>'w/o Temporal Attention Pooling (TAP)'</i> where attention pooling is replaced with mean pooling! This documentation mismatch must be clarified in future publications."""
    
    warn_table = Table([[Paragraph(code_warn, styles['CalloutText'])]], colWidths=[516])
    warn_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#FFFBEB")),
        ('BOX', (0,0), (-1,-1), 1.2, colors.HexColor("#F59E0B")),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(warn_table)

    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: INTERN STRATEGIC ROADMAP & QUICKSTART
    # =========================================================================
    story.append(Paragraph("6. Intern Strategic Roadmap & Concrete Action Plan", styles['SectionHeader']))
    story.append(Paragraph(
        "To elevate ST-GoG from an academic prototype to a publication-grade, production-ready system, execute the following four structured phases:",
        styles['BodyCustom']
    ))

    phases = [
        ("Phase 1: Code Refactoring & Modular Package Setup (Weeks 1 – 2)", [
            "<b>Structure the Repository:</b> Create a standard package: <code>stgog/models/</code>, <code>stgog/data/</code>, <code>stgog/training/</code>, and <code>stgog/utils/</code> with <code>setup.py</code> and <code>requirements.txt</code>.",
            "<b>Hardware-Agnostic Execution:</b> Replace brittle <code>DataParallel</code> batch-slicing logic with device auto-detection that seamlessly supports CPU, single-GPU, and Multi-GPU DistributedDataParallel (DDP).",
            "<b>Config System:</b> Implement YAML configuration files (e.g. <code>configs/default.yaml</code>) to decouple hyperparameters from model logic.",
            "<b>Dataset Caching:</b> Build a caching loader around <code>sktime</code> so benchmark datasets are stored locally rather than redownloaded."
        ]),
        ("Phase 2: Architectural Enhancements & Solving Failure Modes (Weeks 3 – 5)", [
            "<b>Dynamic Hybrid Adjacency:</b> Implement a learnable convex combination of chain topology and correlation: <i>A_hybrid = (1 - &beta;) A_chain + &beta; Softmax((X W_q)(X W_k)^T / sqrt(d))</i>. This directly targets the performance gaps on <i>FaceDetection</i> and <i>DuckDuckGeese</i>.",
            "<b>Strided Temporal Downsampling:</b> Implement a strided pooling or 1D convolution layer prior to GoG segment partitioning for sequences with <i>T > 2,000</i> (e.g. <i>EigenWorms</i>). This will prevent memory explosion and slash training time from hours to minutes.",
            "<b>Multi-Head Attention GoG Readout:</b> Replace segment mean pooling with an attention-weighted pooling module <i>&alpha;_k = Softmax(v^T tanh(W h_k + b))</i> to let the network selectively weight key temporal events."
        ]),
        ("Phase 3: Interpretability, FLOPs Profiling & Evaluation (Weeks 6 – 7)", [
            "<b>Model Explainability (Grad-CAM & Attention Maps):</b> Implement gradient-weighted class activation mapping to visualize which temporal segments and sensor channels triggered classification decisions.",
            "<b>Efficiency Profiling:</b> Compute MACs / GFLOPs, peak memory consumption, and per-sample inference latency using <code>thop</code> to rigorously benchmark efficiency against SAGoG.",
            "<b>Expanded Metrics:</b> Report macro-averaged F1-scores, per-class sensitivity/specificity, and calibration Expected Calibration Error (ECE) across all datasets."
        ]),
        ("Phase 4: Open-Source Release & Conference Manuscript (Weeks 8+)", [
            "<b>Clean GitHub Release:</b> Prepare a clean public repository with reproducible benchmarks, pretrained model weights, and tutorials.",
            "<b>Academic Publication:</b> Format the enhanced architecture, hybrid adjacency results, and efficiency benchmarks into a manuscript targeted at top venues (e.g., IEEE ICASSP, KDD, or NeurIPS Time-Series Workshop)."
        ])
    ]

    for title, items in phases:
        story.append(Paragraph(title, styles['SubSectionHeader']))
        for it in items:
            story.append(Paragraph(f"&bull; {it}", styles['BulletCustom']))
        story.append(Spacer(1, 2))

    story.append(Spacer(1, 6))
    story.append(Paragraph("<b>Quickstart: Running the Model Today</b>", styles['SubSectionHeader']))
    story.append(Paragraph(
        "To run the verified model right now on your machine without notebook overhead:<br/>"
        "1. We have already cleanly extracted <code>d:\\first intern(maybe)\\stgog_cell_7.py</code>.<br/>"
        "2. Edit lines 23-28 to select a lightweight dataset like <code>DATASET_NAME = 'NATOPS'</code> or <code>'FingerMovements'</code>.<br/>"
        "3. Run directly in your terminal: <code>python \"d:\\first intern(maybe)\\stgog_cell_7.py\"</code>.<br/>"
        "The model will train, perform snapshot ensembling, print calibration ECE, and output confusion matrix plots.",
        styles['BodyCustom']
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report PDF built successfully at: {filename}")

if __name__ == "__main__":
    build_pdf()
