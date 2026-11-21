import datetime
"""
GUIDEWISE — CRISPR Educational Simulation Platform
A single-file Streamlit prototype for synthetic sequences only.

Run:
    pip install streamlit plotly
    streamlit run GUIDEWISE_violet_all_modules.py

Safety/educational scope:
- Use only artificial/synthetic example sequences.
- This is a computational teaching simulator, not a laboratory tool.
- It does not predict real editing outcomes or biological off-target activity.
- Sequence similarity and PAM recognition are separate observations.
"""

import difflib
import io
from collections import Counter

import plotly.express as px
import streamlit as st

# Configure the page before rendering any Streamlit elements.
st.set_page_config(page_title="GUIDEWISE | CRISPR Simulator", page_icon="🧬", layout="wide")


# ---------------------------------------------------------------------
# 1. PAGE SETUP AND VISUAL STYLE
# ---------------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #160b2e 0%, #32145c 52%, #542b83 100%) !important;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background: #1b0d35 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(24, 10, 47, 0.75) !important;
    }
    [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }
    h1, h2, h3, p, label, .stMarkdown {
        color: #f3eaff;
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border: 1px solid #24517b;
        border-radius: 18px;
        background: linear-gradient(110deg, #102c4b, #123d60);
        margin-bottom: 1rem;
    }
    .metric-card {
        padding: 1rem;
        border-radius: 14px;
        background: #102945;
        border: 1px solid #24517b;
        min-height: 95px;
    }
    .small-note {
        color: #b8cce2;
        font-size: 0.9rem;
    }
    .sequence {
        font-family: monospace;
        overflow-wrap: anywhere;
        letter-spacing: 1px;
        padding: 0.7rem;
        border-radius: 10px;
        background: #07182b;
        border: 1px solid #21496e;
        color: #bde5ff;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🧬 GUIDEWISE</h1>
      <p><b>CRISPR Educational Simulation Platform</b></p>
      <p class="small-note">Explore guide matching, PAM recognition, sequence differences,
      and hypothetical outcomes using synthetic DNA sequences.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# 2. CONSTANTS, SAMPLE DATA, AND BASIC SEQUENCE UTILITIES
# ---------------------------------------------------------------------

DNA_BASES = set("ACGT")
RNA_BASES = set("ACGU")

SAMPLES = {
    "Exact match + PAM present": {
        "guide": "AUGCCGAUACGUAGCUACGA",
        "dna": "TTTTTCGTAGCTACGTATCGGCATAGGTTT",
        "explanation": "A matching protospacer and an NGG PAM are included in this artificial example.",
    },
    "One mismatch + PAM present": {
        "guide": "AUGCCGAUACGUAGCUACGA",
        "dna": "TTTTACGTAGCTACGTATCGGCATAGGTTT",
        "explanation": "Use the mismatch teaching control to compare one changed guide position.",
    },
    "Multiple mismatches": {
        "guide": "AUGCCGAUACGUAGCUACGA",
        "dna": "TTTTAGTAGCTACGTATCGGCATAGGTTT",
        "explanation": "This example helps demonstrate that sequence differences are counted separately from PAM recognition.",
    },
    "PAM not recognised": {
        "guide": "AUGCCGAUACGUAGCUACGA",
        "dna": "TTTTTCGTAGCTACGTATCGGCATACGTTT",
        "explanation": "The nearby sequence does not contain the expected NGG PAM in the relevant orientation.",
    },
    "Short negative control": {
        "guide": "AUGCCGAUACGUAGCUACGA",
        "dna": "ACGTACGTACGTACGTACGT",
        "explanation": "A short artificial sequence with no complete guide-length target and adjacent PAM.",
    },
}


def clean_sequence(sequence: str) -> str:
    """Remove spaces/newlines and convert a sequence to uppercase."""
    return "".join(sequence.split()).upper()


def validate_dna(sequence: str) -> str:
    """Return a user-friendly validation message; empty string means valid."""
    if not sequence:
        return "DNA sequence is empty."
    invalid = sorted(set(sequence) - DNA_BASES)
    if invalid:
        return "DNA contains invalid character(s): " + ", ".join(invalid)
    return ""


def validate_rna(sequence: str) -> str:
    """Return a user-friendly validation message; empty string means valid."""
    if not sequence:
        return "Guide RNA is empty."
    invalid = sorted(set(sequence) - RNA_BASES)
    if invalid:
        return "Guide RNA must use A, C, G and U (RNA alphabet). Invalid: " + ", ".join(invalid)
    return ""


def reverse_complement(dna: str) -> str:
    """Create the reverse complement of a DNA string."""
    return dna.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def gc_percentage(sequence: str) -> float:
    """Calculate the percentage of G and C bases."""
    if not sequence:
        return 0.0
    return 100.0 * (sequence.count("G") + sequence.count("C")) / len(sequence)


def pam_matches(pam: str, pattern: str = "NGG") -> bool:
    """Check a PAM against a simple pattern where N means any DNA base."""
    if len(pam) != len(pattern):
        return False
    for base, expected in zip(pam, pattern):
        if expected != "N" and base != expected:
            return False
    return True


# ---------------------------------------------------------------------
# 3. GUIDE RNA MATCHING AND PAM SCAN
# ---------------------------------------------------------------------

def scan_targets(dna: str, guide_rna: str, pam_pattern: str = "NGG", max_mismatches: int = 3) -> list:
    """
    Scan both DNA orientations for guide-length windows followed by a PAM.
    Guide RNA is compared with the RNA version of the reverse-complement
    target strand. Coordinates are reported as 1-based positions on the
    original input sequence.
    """
    guide = guide_rna.replace("T", "U")
    guide_length = len(guide)
    hits = []

    orientations = [
        ("Forward strand", dna),
        ("Reverse strand", reverse_complement(dna)),
    ]

    for strand_name, oriented_dna in orientations:
        for start in range(0, len(oriented_dna) - guide_length - len(pam_pattern) + 1):
            protospacer = oriented_dna[start : start + guide_length]
            pam = oriented_dna[start + guide_length : start + guide_length + len(pam_pattern)]
            target_rna = reverse_complement(protospacer).replace("T", "U")

            mismatch_positions = [
                index + 1
                for index, (guide_base, target_base) in enumerate(zip(guide, target_rna))
                if guide_base != target_base
            ]
            mismatch_count = len(mismatch_positions)
            if mismatch_count <= max_mismatches:
                if strand_name == "Forward strand":
                    original_start = start + 1
                    original_end = start + guide_length + len(pam_pattern)
                else:
                    original_start = len(dna) - (start + guide_length + len(pam_pattern)) + 1
                    original_end = len(dna) - start

                hits.append(
                    {
                        "strand": strand_name,
                        "start": original_start,
                        "end": original_end,
                        "protospacer": protospacer,
                        "pam": pam,
                        "pam_ok": pam_matches(pam, pam_pattern),
                        "target_rna": target_rna,
                        "mismatches": mismatch_count,
                        "positions": mismatch_positions,
                    }
                )

    # Prefer fewer mismatches, then PAM-recognised hits.
    hits.sort(key=lambda item: (item["mismatches"], not item["pam_ok"], item["start"]))
    return hits


# ---------------------------------------------------------------------
# 4. MUTATION / SEQUENCE-DIFFERENCE CHECKING
# ---------------------------------------------------------------------

def compare_sequences(before: str, after: str) -> tuple:
    """
    Compare two synthetic sequences using sequence alignment opcodes.
    The output labels substitutions, insertions and deletions.
    Positions are 1-based positions in the before/after sequence context.
    """
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    changes = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        old_part = before[i1:i2]
        new_part = after[j1:j2]

        if tag == "replace":
            paired = min(len(old_part), len(new_part))
            for offset in range(paired):
                changes.append(
                    {
                        "type": "Substitution",
                        "before_position": i1 + offset + 1,
                        "after_position": j1 + offset + 1,
                        "before": old_part[offset],
                        "after": new_part[offset],
                    }
                )
            if len(old_part) > paired:
                changes.append(
                    {
                        "type": "Deletion",
                        "before_position": i1 + paired + 1,
                        "after_position": j1 + paired + 1,
                        "before": old_part[paired:],
                        "after": "—",
                    }
                )
            if len(new_part) > paired:
                changes.append(
                    {
                        "type": "Insertion",
                        "before_position": i1 + paired + 1,
                        "after_position": j1 + paired + 1,
                        "before": "—",
                        "after": new_part[paired:],
                    }
                )

        elif tag == "delete":
            changes.append(
                {
                    "type": "Deletion",
                    "before_position": i1 + 1,
                    "after_position": j1 + 1,
                    "before": old_part,
                    "after": "—",
                }
            )

        elif tag == "insert":
            changes.append(
                {
                    "type": "Insertion",
                    "before_position": i1 + 1,
                    "after_position": j1 + 1,
                    "before": "—",
                    "after": new_part,
                }
            )

    counts = Counter(change["type"] for change in changes)
    return changes, counts


def make_highlighted_comparison(before: str, after: str) -> tuple:
    """Create readable aligned text using SequenceMatcher blocks."""
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    before_parts = []
    after_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old = before[i1:i2]
        new = after[j1:j2]
        if tag == "equal":
            before_parts.append(old)
            after_parts.append(new)
        else:
            before_parts.append(f"[{old or '∅'}]")
            after_parts.append(f"[{new or '∅'}]")
    return "".join(before_parts), "".join(after_parts)


# ---------------------------------------------------------------------
# 5. SIDEBAR NAVIGATION AND INPUT MODES
# ---------------------------------------------------------------------

st.sidebar.title("GUIDEWISE Navigation")
MODULE_GROUPS = {
    "Overview": ["Dashboard"],
    "CRISPR Simulation Lab": [
        "Sequence Simulator",
        "Off-Target Comparison",
        "Hypothetical Editing View",
    ],
    "DNA Analysis Lab": [
        "Nucleotide Composition Analyzer",
        "GC Content Analyzer",
        "Sequence Statistics",
        "DNA Alignment",
        "Mutation Checker",
        "Complementary Strand Generator",
        "Codon Translation",
        "Repeated Sequence Finder",
        "GC Distribution Analyzer",
        "DNA Palindrome Finder",
    ],
    "Genetics & Inheritance Lab": [
        "DNA Relationship Learning",
        "Paternity & Kinship Education",
    ],
    "DNA Repository Lab": ["DNA Storage Repository"],
    "DNA Game Zone": ["DNA Games"],
    "Learning & Reports": [
        "Interactive Quiz",
        "Results & Reports",
        "Biology Learning Hub",
        "About & Limitations",
    ],
}

selected_group = st.sidebar.selectbox("Choose a section", list(MODULE_GROUPS.keys()))
page = st.sidebar.selectbox("Choose a module", MODULE_GROUPS[selected_group])

st.sidebar.markdown("---")
st.sidebar.caption("Synthetic educational simulation only.")
input_mode = st.sidebar.radio("Sequence source", ["Sample dataset", "Manual input"])


# ---------------------------------------------------------------------
# 6. SHARED INPUT HANDLING
# ---------------------------------------------------------------------

if input_mode == "Sample dataset":
    selected_sample = st.sidebar.selectbox("Choose a sample", list(SAMPLES.keys()))
    sample = SAMPLES[selected_sample]
    default_guide = sample["guide"]
    default_dna = sample["dna"]
else:
    selected_sample = None
    default_guide = "AUGCCGAUACGUAGCUACGA"
    default_dna = "TTTTCGTAGCTACGTATCGGCATACGATTT"

if page in ["Dashboard", "Sequence Simulator"]:
    st.sidebar.markdown("### Input sequences")
    guide_input = st.sidebar.text_area("Guide RNA (A, C, G, U)", value=default_guide, height=90)
    dna_input = st.sidebar.text_area("Synthetic DNA (A, C, G, T)", value=default_dna, height=110)
    pam_pattern = st.sidebar.selectbox("PAM pattern", ["NGG", "NAG", "NGA"])
    max_mismatches = st.sidebar.slider("Maximum mismatches to display", 0, 10, 5)
else:
    guide_input = default_guide
    dna_input = default_dna
    pam_pattern = "NGG"
    max_mismatches = 5

guide = clean_sequence(guide_input)
dna = clean_sequence(dna_input)
guide_error = validate_rna(guide)
dna_error = validate_dna(dna)


# ---------------------------------------------------------------------
# 7. DASHBOARD
# ---------------------------------------------------------------------

if page == "Dashboard":
    st.subheader("Project Overview")
    st.write(
        "GUIDEWISE demonstrates how a guide RNA can be compared with artificial DNA "
        "segments, how a selected PAM pattern is checked, and how two sequences can "
        "be compared for substitutions, insertions and deletions."
    )

    total_modules = sum(len(mods) for mods in MODULE_GROUPS.values())
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Learning modules", str(total_modules))
    with col2:
        st.metric("Sequence source", input_mode)
    with col3:
        st.metric("PAM example", pam_pattern)
    with col4:
        st.metric("Scope", "Synthetic")

    st.markdown("### Included modules")
    modules = [
        ("1. Main Dashboard", "Quick overview and navigation."),
        ("2. Sequence Input", "Manual entry or ready-made synthetic test cases."),
        ("3. Guide RNA Matching", "Compare guide RNA against both sequence orientations."),
        ("4. PAM Recognition", "Check PAM separately from sequence similarity."),
        ("5. Target Region Visualisation", "Show candidate protospacer and adjacent PAM regions."),
        ("6. Off-Target Comparison", "Display similar sequence regions without claiming biological activity."),
        ("7. Additional Learning Factors", "GC content, guide length, Cas protein and chromatin concepts."),
        ("8. Mutation Checker", "Compare before/after strings for substitutions, insertions and deletions."),
        ("9. Hypothetical Editing View", "Illustrative before/after display only."),
        ("10. Results & Reports", "Tables, charts and downloadable educational summary."),
    ]
    for name, description in modules:
        with st.expander(name):
            st.write(description)

    st.info(
        "Start with **Sequence Simulator** to run guide matching and PAM checks. "
        "Use **Mutation Checker** for sequence-difference analysis."
    )


# ---------------------------------------------------------------------
# 8. SEQUENCE SIMULATOR
# ---------------------------------------------------------------------

elif page == "Sequence Simulator":
    st.subheader("Guide RNA Matching & PAM Recognition")

    if guide_error or dna_error:
        if guide_error:
            st.error(guide_error)
        if dna_error:
            st.error(dna_error)
        st.stop()

    if len(guide) != 20:
        st.warning(
            f"This teaching configuration expects a 20-base guide. Your guide has {len(guide)} bases. "
            "The simulator will still compare the entered length."
        )

    hits = scan_targets(dna, guide, pam_pattern, max_mismatches)

    col1, col2, col3 = st.columns(3)
    col1.metric("Guide length", f"{len(guide)} nt")
    col2.metric("Guide GC content", f"{gc_percentage(guide):.1f}%")
    col3.metric("Candidate regions", len(hits))

    st.markdown("### Input preview")
    st.markdown(f'<div class="sequence"><b>Guide RNA:</b> {guide}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sequence"><b>DNA:</b> {dna}</div>', unsafe_allow_html=True)

    st.markdown("### Candidate target regions")
    if not hits:
        st.warning(
            "No candidate region was found within the selected mismatch limit. "
            "This does not establish that no biological target exists."
        )
    else:
        rows = []
        for number, hit in enumerate(hits, start=1):
            rows.append(
                {
                    "Candidate": number,
                    "Strand": hit["strand"],
                    "DNA positions": f'{hit["start"]}–{hit["end"]}',
                    "Protospacer": hit["protospacer"],
                    "PAM": hit["pam"],
                    "PAM recognised": "Yes" if hit["pam_ok"] else "No",
                    "Mismatches": hit["mismatches"],
                    "Mismatch positions": ", ".join(map(str, hit["positions"])) if hit["positions"] else "None",
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

        st.markdown("### Candidate comparison chart")
        chart_data = [
            {"Candidate": f"Region {i + 1}", "Mismatch count": hit["mismatches"]}
            for i, hit in enumerate(hits)
        ]
        fig = px.bar(chart_data, x="Candidate", y="Mismatch count", title="Mismatch count per candidate")
        st.plotly_chart(fig, width="stretch")

        best = hits[0]
        st.markdown("### Best sequence-similarity candidate")
        st.write(f"**Strand:** {best['strand']}  |  **Mismatch count:** {best['mismatches']}")
        st.write(f"**PAM:** {best['pam']} — {'recognised' if best['pam_ok'] else 'not recognised'}")
        st.write(f"**Mismatch positions (guide positions):** {best['positions'] or 'None'}")
        if best["pam_ok"] and best["mismatches"] == 0:
            st.success("This synthetic example has an exact sequence match and a recognised PAM pattern.")
        elif best["pam_ok"]:
            st.info("A PAM pattern is present, but the guide comparison contains mismatches.")
        else:
            st.warning("Sequence similarity was found, but this candidate does not pass the selected PAM pattern check.")

    st.markdown("### Hypothetical editing visualisation")
    st.caption("Illustration only: this does not perform or predict real genetic editing.")
    if hits:
        selected_hit = hits[0]
        st.write("Candidate protospacer:")
        st.code(selected_hit["protospacer"])
        st.write("Adjacent PAM:")
        st.code(selected_hit["pam"])
        st.write("Illustrative marker:  [POTENTIAL CUT REGION] — shown for learning, not as an experimental result.")
    else:
        st.write("No candidate region is available to illustrate for the current settings.")

    st.markdown("### Results report")
    report = io.StringIO()
    report.write("GUIDEWISE — SYNTHETIC EDUCATIONAL REPORT\n")
    report.write(f"Guide RNA: {guide}\nDNA: {dna}\nPAM pattern: {pam_pattern}\n")
    report.write("All results are computational teaching comparisons only.\n\n")
    for i, hit in enumerate(hits, start=1):
        report.write(
            f"Candidate {i}: {hit['strand']}, positions {hit['start']}-{hit['end']}, "
            f"PAM={hit['pam']} recognised={hit['pam_ok']}, mismatches={hit['mismatches']}, "
            f"mismatch_positions={hit['positions']}\n"
        )
    st.download_button(
        "Download simulation report",
        data=report.getvalue(),
        file_name="guidewise_simulation_report.txt",
        mime="text/plain",
    )


# ---------------------------------------------------------------------
# 9. MUTATION CHECKER
# ---------------------------------------------------------------------

elif page == "Mutation Checker":
    st.subheader("Mutation Checking Module")
    st.write(
        "Compare two artificial DNA strings and identify sequence differences. "
        "This module reports computational string differences; it is not a clinical "
        "variant caller or a predictor of real biological mutations."
    )

    before_default = "ATGCCGTTAACCGGTT"
    after_default = "ATGTCGTTAAACCGGTT"

    before_text = st.text_area("Before sequence (synthetic DNA)", before_default, height=90)
    after_text = st.text_area("After sequence (synthetic DNA)", after_default, height=90)

    before = clean_sequence(before_text)
    after = clean_sequence(after_text)

    before_error = validate_dna(before)
    after_error = validate_dna(after)

    if before_error:
        st.error("Before sequence: " + before_error)
    if after_error:
        st.error("After sequence: " + after_error)

    if not before_error and not after_error:
        changes, counts = compare_sequences(before, after)
        substitutions = counts.get("Substitution", 0)
        insertions = counts.get("Insertion", 0)
        deletions = counts.get("Deletion", 0)

        st.markdown("### Difference summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total difference events", len(changes))
        c2.metric("Substitutions", substitutions)
        c3.metric("Insertions", insertions)
        c4.metric("Deletions", deletions)

        st.markdown("### Before-and-after sequence view")
        highlighted_before, highlighted_after = make_highlighted_comparison(before, after)
        st.write("Bracketed sections indicate regions that differ.")
        st.markdown(f'<div class="sequence"><b>Before:</b> {highlighted_before}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sequence"><b>After:</b> {highlighted_after}</div>', unsafe_allow_html=True)

        st.markdown("### Difference positions and types")
        if changes:
            st.dataframe(changes, width="stretch", hide_index=True)
            chart_data = [
                {"Difference type": kind, "Count": counts.get(kind, 0)}
                for kind in ["Substitution", "Insertion", "Deletion"]
            ]
            fig = px.bar(chart_data, x="Difference type", y="Count", title="Difference type summary")
            st.plotly_chart(fig, width="stretch")
        else:
            st.success("The two synthetic sequences are identical. No differences were detected.")

        st.markdown("### Educational interpretation")
        if not changes:
            st.info("Both strings are identical.")
        else:
            st.write(
                f"The comparison found {substitutions} substitution event(s), "
                f"{insertions} insertion event(s), and {deletions} deletion event(s)."
            )
            st.caption(
                "Alignment can be ambiguous when repeated bases occur. Treat these as "
                "teaching-oriented sequence differences, not confirmed biological mutation calls."
            )

        mutation_report = io.StringIO()
        mutation_report.write("GUIDEWISE — SYNTHETIC SEQUENCE DIFFERENCE REPORT\n")
        mutation_report.write(f"Before: {before}\nAfter: {after}\n\n")
        mutation_report.write(f"Substitutions: {substitutions}\nInsertions: {insertions}\nDeletions: {deletions}\n\n")
        for change in changes:
            mutation_report.write(str(change) + "\n")
        st.download_button(
            "Download mutation comparison report",
            data=mutation_report.getvalue(),
            file_name="guidewise_mutation_report.txt",
            mime="text/plain",
        )


# ---------------------------------------------------------------------
# 10. BONUS: BIOLOGICAL LEARNING HUB
# ---------------------------------------------------------------------

elif page == "Biology Learning Hub":
    st.subheader("Bonus Features — Biological Learning Hub")

    st.markdown("### GC content calculator")
    gc_sequence = st.text_input("Enter a synthetic DNA or RNA sequence", "ATGCGCATTA")
    gc_clean = clean_sequence(gc_sequence)
    if gc_clean and not (set(gc_clean) - (DNA_BASES | RNA_BASES)):
        st.metric("GC content", f"{gc_percentage(gc_clean):.2f}%")
        st.progress(min(gc_percentage(gc_clean) / 100.0, 1.0))
    else:
        st.warning("Enter only A, C, G, T or U.")

    st.markdown("### Guide RNA characteristics")
    guide_for_info = st.text_input("Guide RNA for learning summary", "AUGCCGAUACGUAGCUACGA")
    guide_for_info = clean_sequence(guide_for_info)
    if not validate_rna(guide_for_info):
        st.write(f"- Length: **{len(guide_for_info)} nucleotides**")
        st.write(f"- GC content: **{gc_percentage(guide_for_info):.1f}%**")
        st.write("- RNA alphabet check: **Valid**")
        st.write("- Sequence is treated as an educational input; no efficacy or safety prediction is made.")
    else:
        st.warning(validate_rna(guide_for_info))

    st.markdown("### Cas protein information")
    cas_choice = st.selectbox("Choose a Cas protein concept", ["SpCas9", "SaCas9", "Cas12a (Cpf1)"])
    cas_info = {
        "SpCas9": "A commonly taught Cas9 example. In this simulator, NGG is used as the default illustrative PAM pattern.",
        "SaCas9": "A different Cas9 protein with different recognition requirements. This app does not model its full sequence rules.",
        "Cas12a (Cpf1)": "A distinct CRISPR-associated nuclease family with different PAM orientation and recognition preferences. Not simulated by the current matching engine.",
    }
    st.info(cas_info[cas_choice])

    st.markdown("### Chromatin accessibility — concept")
    st.write(
        "In cells, DNA is packaged with proteins into chromatin. Some regions may be more "
        "accessible than others. Accessibility can influence whether molecular machinery "
        "can reach a DNA region. GUIDEWISE does not measure chromatin state and does not "
        "infer accessibility from the entered sequence."
    )
    st.markdown("### Sequence similarity vs. biological activity")
    st.write(
        "- **Sequence similarity:** how closely two strings match.\n"
        "- **PAM recognition:** whether the selected PAM text pattern is present beside a candidate region.\n"
        "- **Off-target possibility:** a teaching concept that motivates careful comparison.\n"
        "- **Actual off-target activity:** a biological outcome that cannot be established by this simulator."
    )


# ---------------------------------------------------------------------
# 11. ADDITIONAL LAB MODULES
# ---------------------------------------------------------------------

elif page == "Off-Target Comparison":
    st.subheader("Educational Sequence Similarity Comparison")
    st.info("Use only invented DNA strings. Similarity scores are string comparisons, not predictions of biological off-target activity.")
    target = clean_sequence(st.text_input("Reference synthetic DNA sequence", "ATGCGTAC"))
    candidates_text = st.text_area("Candidate synthetic DNA sequences (one per line)", "ATGCGTAC\nATGAGTAC\nTTGCGTCC")
    if validate_dna(target):
        st.error(validate_dna(target))
    else:
        rows = []
        for number, raw in enumerate(candidates_text.splitlines(), 1):
            candidate = clean_sequence(raw)
            if not candidate:
                continue
            if validate_dna(candidate):
                st.warning(f"Candidate {number} contains invalid DNA letters and was skipped.")
                continue
            matcher = difflib.SequenceMatcher(a=target, b=candidate, autojunk=False)
            similarity = matcher.ratio() * 100
            mismatches = sum(a != b for a, b in zip(target, candidate)) + abs(len(target) - len(candidate))
            rows.append(
                {
                    "Candidate": f"Candidate {number}",
                    "Sequence": candidate,
                    "Length": len(candidate),
                    "Position-wise differences*": mismatches,
                    "String similarity": f"{similarity:.1f}%",
                }
            )
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            st.caption("*Simple position-wise difference count plus length difference; not a biological mismatch model.")
            chosen = st.selectbox("Show detailed comparison", [r["Candidate"] for r in rows])
            candidate = next(r["Sequence"] for r in rows if r["Candidate"] == chosen)
            matcher = difflib.SequenceMatcher(a=target, b=candidate, autojunk=False)
            st.write("Reference:", target)
            st.write("Candidate:", candidate)
            st.write(f"Similarity: **{matcher.ratio() * 100:.1f}%**")
        else:
            st.warning("Enter at least one valid synthetic candidate sequence.")

elif page == "Hypothetical Editing View":
    st.subheader("Hypothetical Sequence Change Viewer")
    st.info("This is a visual teaching example only. It does not edit DNA or model a real CRISPR outcome.")
    demo = clean_sequence(st.text_input("Synthetic DNA sequence", "ATGCGTAC"))
    if validate_dna(demo):
        st.error(validate_dna(demo))
    elif demo:
        if len(demo) == 1:
            position = 1
            st.write("Display position: **1** (sequence has 1 base)")
        else:
            position = st.number_input(
                "Choose a display position (1-based)",
                min_value=1,
                max_value=len(demo),
                value=1,
                step=1,
            )
        replacement = st.selectbox(
            "Choose a different display letter",
            [b for b in "ATGC" if b != demo[int(position) - 1]],
        )
        illustrative = demo[: int(position) - 1] + replacement + demo[int(position) :]
        st.markdown("**Before (synthetic example)**")
        st.code(demo)
        st.markdown("**Illustrative after (one character changed for learning)**")
        st.code(illustrative)
        st.caption("The changed letter is a made-up demonstration, not a recommended edit or predicted biological result.")

elif page == "Nucleotide Composition Analyzer":
    st.subheader("Nucleotide Composition Analyzer")
    seq = clean_sequence(st.text_area("Enter synthetic DNA sequence", "ATGCGCATTA"))
    if validate_dna(seq):
        st.error(validate_dna(seq))
    else:
        counts = {base: seq.count(base) for base in "ATGC"}
        total = len(seq)
        st.metric("Total nucleotides", total)
        comp_rows = [
            {"Nucleotide": b, "Count": counts[b], "Percentage": round(counts[b] / total * 100, 2)}
            for b in "ATGC"
        ]
        st.dataframe(comp_rows, width="stretch", hide_index=True)
        fig = px.bar(comp_rows, x="Nucleotide", y="Count", color="Nucleotide", title="Nucleotide counts")
        st.plotly_chart(fig, width="stretch")
        fig2 = px.pie(comp_rows, names="Nucleotide", values="Count", title="Nucleotide composition")
        st.plotly_chart(fig2, width="stretch")

elif page == "GC Content Analyzer":
    st.subheader("GC Content Analyzer")
    seq = clean_sequence(st.text_area("Enter synthetic DNA sequence", "ATGCGCATTA"))
    if validate_dna(seq):
        st.error(validate_dna(seq))
    else:
        gc = gc_percentage(seq)
        st.metric("GC content", f"{gc:.2f}%")
        st.progress(min(gc / 100.0, 1.0))
        st.write(f"G bases: **{seq.count('G')}** | C bases: **{seq.count('C')}**")
        st.write(f"A bases: **{seq.count('A')}** | T bases: **{seq.count('T')}**")

elif page == "Sequence Statistics":
    st.subheader("Sequence Statistics")
    seq = clean_sequence(st.text_area("Enter synthetic DNA sequence", "ATGCGCATTA"))
    if validate_dna(seq):
        st.error(validate_dna(seq))
    else:
        cols = st.columns(3)
        cols[0].metric("Length", len(seq))
        cols[1].metric("GC content", f"{gc_percentage(seq):.1f}%")
        cols[2].metric("AT content", f"{100 - gc_percentage(seq):.1f}%")
        st.write({base: seq.count(base) for base in "ATGC"})

elif page == "DNA Alignment":
    st.subheader("DNA Sequence Alignment")
    first = clean_sequence(st.text_input("First synthetic sequence", "ATGCGTAC"))
    second = clean_sequence(st.text_input("Second synthetic sequence", "ATGAGTAC"))
    if validate_dna(first) or validate_dna(second):
        st.error(validate_dna(first) or validate_dna(second))
    else:
        matcher = difflib.SequenceMatcher(a=first, b=second, autojunk=False)
        st.metric("Sequence similarity", f"{matcher.ratio() * 100:.1f}%")
        st.write("First:")
        st.code(first)
        st.write("Second:")
        st.code(second)
        diff_rows = []
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                diff_rows.append(
                    {
                        "Change": tag.title(),
                        "First positions": f"{i1 + 1}-{i2}",
                        "Second positions": f"{j1 + 1}-{j2}",
                        "First bases": first[i1:i2] or "—",
                        "Second bases": second[j1:j2] or "—",
                    }
                )
        if diff_rows:
            st.dataframe(diff_rows, width="stretch", hide_index=True)
        else:
            st.success("The two strings are identical.")

elif page == "Complementary Strand Generator":
    st.subheader("Complementary Strand Generator")
    seq = clean_sequence(st.text_input("Synthetic DNA sequence", "ATGCGTAC"))
    if validate_dna(seq):
        st.error(validate_dna(seq))
    else:
        complement = seq.translate(str.maketrans("ATGC", "TACG"))
        st.write("Original sequence")
        st.code(seq)
        st.write("Complementary sequence")
        st.code(complement)
        st.write("Reverse complement")
        st.code(complement[::-1])

elif page == "Codon Translation":
    st.subheader("Codon Translation Learning Tool")
    st.info(
        "Enter an invented coding DNA example in the 5′ to 3′ direction. "
        "This simplified tool reads triplets from the first base; it does not identify real genes."
    )
    seq = clean_sequence(st.text_input("Synthetic coding DNA sequence", "ATGGCTTAA"))
    codon_table = {
        "TTT": "Phe", "TTC": "Phe", "TTA": "Leu", "TTG": "Leu",
        "TCT": "Ser", "TCC": "Ser", "TCA": "Ser", "TCG": "Ser",
        "TAT": "Tyr", "TAC": "Tyr", "TAA": "Stop", "TAG": "Stop",
        "TGT": "Cys", "TGC": "Cys", "TGA": "Stop", "TGG": "Trp",
        "CTT": "Leu", "CTC": "Leu", "CTA": "Leu", "CTG": "Leu",
        "CCT": "Pro", "CCC": "Pro", "CCA": "Pro", "CCG": "Pro",
        "CAT": "His", "CAC": "His", "CAA": "Gln", "CAG": "Gln",
        "CGT": "Arg", "CGC": "Arg", "CGA": "Arg", "CGG": "Arg",
        "ATT": "Ile", "ATC": "Ile", "ATA": "Ile", "ATG": "Met",
        "ACT": "Thr", "ACC": "Thr", "ACA": "Thr", "ACG": "Thr",
        "AAT": "Asn", "AAC": "Asn", "AAA": "Lys", "AAG": "Lys",
        "AGT": "Ser", "AGC": "Ser", "AGA": "Arg", "AGG": "Arg",
        "GTT": "Val", "GTC": "Val", "GTA": "Val", "GTG": "Val",
        "GCT": "Ala", "GCC": "Ala", "GCA": "Ala", "GCG": "Ala",
        "GAT": "Asp", "GAC": "Asp", "GAA": "Glu", "GAG": "Glu",
        "GGT": "Gly", "GGC": "Gly", "GGA": "Gly", "GGG": "Gly",
    }
    if validate_dna(seq):
        st.error(validate_dna(seq))
    else:
        codons = [seq[i : i + 3] for i in range(0, len(seq) - 2, 3)]
        translated = [codon_table[c] for c in codons]
        st.write("Codons:", " | ".join(codons) if codons else "None")
        st.write("Amino-acid labels:", " – ".join(translated) if translated else "None")
        if len(seq) % 3:
            st.caption(f"The final {len(seq) % 3} base(s) do not form a complete codon and were not translated.")

elif page == "Repeated Sequence Finder":
    st.subheader("Repeated Sequence Finder")
    seq = clean_sequence(st.text_input("Synthetic DNA sequence", "ATGATGCCATGATG"))
    k = st.slider("Pattern length", 2, 8, 3)
    if validate_dna(seq):
        st.error(validate_dna(seq))
    else:
        counts = Counter(seq[i : i + k] for i in range(max(0, len(seq) - k + 1)))
        repeated = [{"Pattern": p, "Occurrences": n} for p, n in counts.items() if n > 1]
        if repeated:
            st.dataframe(
                sorted(repeated, key=lambda x: (-x["Occurrences"], x["Pattern"])),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No repeated patterns of this length were found.")

elif page == "GC Distribution Analyzer":
    st.subheader("GC Distribution Analyzer")
    seq = clean_sequence(st.text_input("Synthetic DNA sequence", "ATGCGCATTAATGCGC"))
    if validate_dna(seq):
        st.error(validate_dna(seq))
    elif len(seq) < 5:
        st.warning("Enter a DNA sequence with at least 5 bases to slide a window across it.")
    else:
        max_win = min(50, len(seq))
        default_win = min(6, max_win)
        window = st.slider("Window size", 4, max_win, default_win)
        points = [
            {"Start position": i + 1, "GC percentage": gc_percentage(seq[i : i + window])}
            for i in range(len(seq) - window + 1)
        ]
        st.plotly_chart(
            px.line(points, x="Start position", y="GC percentage", title=f"GC content across {window}-base windows"),
            width="stretch",
        )

elif page == "DNA Palindrome Finder":
    st.subheader("DNA Reverse-Complement Palindrome Finder")
    seq = clean_sequence(st.text_input("Synthetic DNA sequence", "ATGCAT"))
    min_len = st.slider("Minimum pattern length", 4, 12, 4, step=2)
    if validate_dna(seq):
        st.error(validate_dna(seq))
    else:
        found = set()
        for size in range(min_len, min(len(seq), 30) + 1, 2):
            for i in range(len(seq) - size + 1):
                part = seq[i : i + size]
                rev_comp = part.translate(str.maketrans("ATGC", "TACG"))[::-1]
                if part == rev_comp:
                    found.add((i + 1, part))
        if found:
            st.dataframe(
                [{"Start": p, "Pattern": s, "Length": len(s)} for p, s in sorted(found)],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No reverse-complement palindromic patterns were found.")

elif page == "DNA Relationship Learning":
    st.subheader("DNA Relationship Learning")
    st.info("Fictional inheritance example. This teaches that a child receives genetic information from both biological parents.")
    st.write("Example marker alleles (fictional):")
    st.table(
        [
            {"Marker": "M1", "Parent A": "10 / 12", "Parent B": "14 / 16", "Child": "12 / 14"},
            {"Marker": "M2", "Parent A": "8 / 9", "Parent B": "9 / 11", "Child": "8 / 11"},
            {"Marker": "M3", "Parent A": "15 / 17", "Parent B": "13 / 15", "Child": "17 / 13"},
        ]
    )
    st.caption("This is a made-up teaching example, not a real relationship assessment.")

elif page == "Paternity & Kinship Education":
    st.subheader("Paternity & Kinship Education")
    st.warning("Educational demonstration only. Do not enter real people's genetic data. This tool cannot determine paternity or biological relationships.")
    st.write("A laboratory may compare many validated genetic markers and interpret them using specialist methods.")
    st.table(
        [
            {"Fictional marker": "Marker A", "Person 1": "10 / 12", "Child": "12 / 14", "Teaching note": "One allele is shared"},
            {"Fictional marker": "Marker B", "Person 1": "8 / 9", "Child": "8 / 11", "Teaching note": "One allele is shared"},
            {"Fictional marker": "Marker C", "Person 1": "15 / 17", "Child": "13 / 15", "Teaching note": "One allele is shared"},
        ]
    )
    st.caption("Shared markers alone do not prove a relationship. Real testing requires validated laboratory analysis and consent.")

elif page == "DNA Storage Repository":
    st.subheader("Synthetic DNA Storage Repository")
    st.info(
        "Store, organize, search, and export your synthetic DNA and RNA sequences associated with Person 1 or Person 2 "
        "along with exact timestamps during this session. Only artificial/synthetic teaching sequences should be stored."
    )

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    if "dna_repository" not in st.session_state:
        st.session_state.dna_repository = [
            {
                "ID": "SEQ-001",
                "Name": "Synthetic Protospacer Alpha",
                "Person": "Person 1",
                "Type": "DNA",
                "Sequence": "TTTTTCGTAGCTACGTATCGGCATAGGTTT",
                "Length": 30,
                "GC %": round(gc_percentage("TTTTTCGTAGCTACGTATCGGCATAGGTTT"), 2),
                "Date & Time": "2026-06-01 10:30",
                "Tag": "CRISPR Demo",
                "Notes": "Artificial target with NGG PAM",
            },
            {
                "ID": "SEQ-002",
                "Name": "Synthetic Guide RNA-1",
                "Person": "Person 2",
                "Type": "RNA",
                "Sequence": "AUGCCGAUACGUAGCUACGA",
                "Length": 20,
                "GC %": round(gc_percentage("AUGCCGAUACGUAGCUACGA"), 2),
                "Date & Time": "2026-06-02 14:15",
                "Tag": "Guide RNA",
                "Notes": "20-nt synthetic teaching guide",
            },
            {
                "ID": "SEQ-003",
                "Name": "GC-Rich Control Fragment",
                "Person": "Person 1",
                "Type": "DNA",
                "Sequence": "GCGCATCGATCGCCGGATGCGC",
                "Length": 22,
                "GC %": round(gc_percentage("GCGCATCGATCGCCGGATGCGC"), 2),
                "Date & Time": "2026-06-03 09:00",
                "Tag": "Control",
                "Notes": "High-GC artificial sequence for composition testing",
            },
        ]

    tab_browse, tab_add, tab_manage = st.tabs(["Browse & Search", "Add Sequence", "Manage & Export"])

    with tab_browse:
        repo = st.session_state.dna_repository
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Stored sequences", len(repo))
        c2.metric("Person 1 entries", sum(1 for item in repo if item["Person"] == "Person 1"))
        c3.metric("Person 2 entries", sum(1 for item in repo if item["Person"] == "Person 2"))
        c4.metric("DNA / RNA", f"{sum(1 for i in repo if i['Type']=='DNA')} / {sum(1 for i in repo if i['Type']=='RNA')}")

        col_search, col_person_filter, col_type_filter = st.columns([2, 1, 1])
        search_query = col_search.text_input("Search by name, ID, tag, or motif", "").strip().upper()
        person_filter = col_person_filter.selectbox("Filter by person", ["All", "Person 1", "Person 2"])
        type_filter = col_type_filter.selectbox("Filter by type", ["All", "DNA", "RNA"])

        filtered_repo = [
            item
            for item in repo
            if (person_filter == "All" or item["Person"] == person_filter)
            and (type_filter == "All" or item["Type"] == type_filter)
            and (
                not search_query
                or search_query in item["ID"].upper()
                or search_query in item["Name"].upper()
                or search_query in item["Person"].upper()
                or search_query in item["Tag"].upper()
                or search_query in item["Sequence"].upper()
                or search_query in item["Notes"].upper()
            )
        ]

        if filtered_repo:
            st.dataframe(filtered_repo, width="stretch", hide_index=True)

            selected_id = st.selectbox(
                "Inspect sequence record",
                [f"{item['ID']} — {item['Name']} ({item['Person']})" for item in filtered_repo],
            )
            chosen_id = selected_id.split(" — ")[0]
            record = next(item for item in filtered_repo if item["ID"] == chosen_id)

            st.markdown(f"**{record['Name']}** (`{record['ID']}` | **{record['Person']}** | `{record['Type']}` | Tag: `{record['Tag']}` | Stored: `{record['Date & Time']}`)")
            st.markdown(f'<div class="sequence">{record["Sequence"]}</div>', unsafe_allow_html=True)
            st.write(f"**Length:** {record['Length']} nt  |  **GC Content:** {record['GC %']:.2f}%")
            if record["Type"] == "DNA":
                st.write(f"**Reverse Complement:** `{reverse_complement(record['Sequence'])}`")
            if record["Notes"]:
                st.caption(f"Notes: {record['Notes']}")
        else:
            st.warning("No stored sequences match your current filter.")

    with tab_add:
        with st.form("add_sequence_form", clear_on_submit=True):
            seq_name = st.text_input("Sequence name", placeholder="e.g., Synthetic Target Beta")
            col_person, col_type, col_tag = st.columns(3)
            seq_person = col_person.selectbox("Assigned Person", ["Person 1", "Person 2"])
            seq_type = col_type.selectbox("Sequence type", ["DNA", "RNA"])
            seq_tag = col_tag.text_input("Category / Tag", value="Synthetic")
            raw_seq = st.text_area("Sequence string (A, C, G, T for DNA; A, C, G, U for RNA)", height=90)
            seq_notes = st.text_input("Educational notes (optional)")
            submitted = st.form_submit_button("Save to Repository")

            if submitted:
                cleaned = clean_sequence(raw_seq)
                err = validate_dna(cleaned) if seq_type == "DNA" else validate_rna(cleaned)
                if not seq_name.strip():
                    st.error("Please enter a sequence name.")
                elif err:
                    st.error(err)
                else:
                    next_num = len(st.session_state.dna_repository) + 1
                    existing_ids = {item["ID"] for item in st.session_state.dna_repository}
                    while f"SEQ-{next_num:03d}" in existing_ids:
                        next_num += 1
                    new_entry = {
                        "ID": f"SEQ-{next_num:03d}",
                        "Name": seq_name.strip(),
                        "Person": seq_person,
                        "Type": seq_type,
                        "Sequence": cleaned,
                        "Length": len(cleaned),
                        "GC %": round(gc_percentage(cleaned), 2),
                        "Date & Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "Tag": seq_tag.strip() or "Synthetic",
                        "Notes": seq_notes.strip(),
                    }
                    st.session_state.dna_repository.append(new_entry)
                    st.success(f"Saved '{new_entry['Name']}' ({new_entry['ID']} for {new_entry['Person']} at {new_entry['Date & Time']}) to the repository.")

    with tab_manage:
        repo = st.session_state.dna_repository
        if not repo:
            st.info("The repository is currently empty.")
        else:
            delete_choice = st.selectbox(
                "Select a sequence to delete",
                [f"{item['ID']} — {item['Name']} ({item['Person']})" for item in repo],
            )
            if st.button("Delete selected sequence"):
                del_id = delete_choice.split(" — ")[0]
                st.session_state.dna_repository = [item for item in repo if item["ID"] != del_id]
                st.success(f"Deleted {del_id}.")
                st.rerun()

            st.markdown("### Export Repository")
            csv_lines = ["ID,Name,Person,Type,Sequence,Length,GC_Percent,DateTime,Tag,Notes"]
            fasta_lines = []
            for item in st.session_state.dna_repository:
                safe_name = item["Name"].replace('"', '""')
                safe_tag = item["Tag"].replace('"', '""')
                safe_notes = item["Notes"].replace('"', '""')
                csv_lines.append(
                    f'{item["ID"]},"{safe_name}",{item["Person"]},{item["Type"]},{item["Sequence"]},'
                    f'{item["Length"]},{item["GC %"]},"{item["Date & Time"]}","{safe_tag}","{safe_notes}"'
                )
                fasta_lines.append(f">{item['ID']} | {item['Name']} | {item['Person']} | {item['Type']} | {item['Date & Time']} | {item['Tag']}")
                fasta_lines.append(item["Sequence"])

            col_dl1, col_dl2 = st.columns(2)
            col_dl1.download_button(
                "Download Repository (CSV)",
                data="\n".join(csv_lines),
                file_name="GUIDEWISE_DNA_repository.csv",
                mime="text/csv",
            )
            col_dl2.download_button(
                "Download Repository (FASTA)",
                data="\n".join(fasta_lines),
                file_name="GUIDEWISE_DNA_repository.fasta",
                mime="text/plain",
            )

elif page == "DNA Games":
    st.subheader("DNA Game Zone")
    game = st.selectbox(
        "Choose a game",
        [
            "DNA Base Pairing Challenge",
            "Find the Mutation",
            "DNA Sequence Puzzle",
            "Codon Matching Game",
            "DNA Structure Identification",
        ],
    )
    if game == "DNA Base Pairing Challenge":
        st.write("Choose the complementary base for each DNA base.")
        questions = [("A", "T"), ("T", "A"), ("G", "C"), ("C", "G")]
        answers = []
        for i, (base, correct) in enumerate(questions):
            ans = st.selectbox(f"{base} pairs with", ["Choose...", "A", "T", "G", "C"], key=f"pair{i}")
            answers.append((ans, correct))
        if st.button("Check base-pairing answers"):
            score = sum(a == c for a, c in answers)
            st.success(f"Your score: {score}/4")

    elif game == "Find the Mutation":
        st.write("Compare these two fictional DNA strings.")
        st.code("ATGCGTAC\nATGAGTAC")
        answer = st.radio("Which position differs?", ["Position 2", "Position 4", "Position 6", "No difference"])
        if st.button("Check mutation answer"):
            st.success("Correct!" if answer == "Position 4" else "Try again. Compare the strings one letter at a time.")

    elif game == "DNA Sequence Puzzle":
        st.write("Arrange these shuffled bases into the displayed target sequence.")
        target_word = "ATGC"
        shuffled = st.text_input("Enter the bases in the correct order", "TGAC")
        if st.button("Check sequence puzzle"):
            st.success(
                "Correct sequence!"
                if clean_sequence(shuffled) == target_word
                else "Not quite. Hint: A comes first, followed by T, G and C."
            )

    elif game == "Codon Matching Game":
        codon_answer = st.selectbox(
            "Which amino-acid label corresponds to AUG in RNA?",
            ["Choose...", "Methionine (Met)", "Glycine (Gly)", "Tryptophan (Trp)"],
        )
        if st.button("Check codon answer"):
            st.success(
                "Correct!"
                if codon_answer == "Methionine (Met)"
                else "Try again. AUG is the usual start codon and codes for methionine."
            )

    elif game == "DNA Structure Identification":
        structure_answer = st.selectbox(
            "Which bases pair in DNA?",
            ["Choose...", "A–T and G–C", "A–G and T–C", "A–C and G–T"],
        )
        if st.button("Check structure answer"):
            st.success(
                "Correct!"
                if structure_answer == "A–T and G–C"
                else "Try again. Remember A pairs with T, and G pairs with C."
            )

elif page == "Interactive Quiz":
    st.subheader("CRISPR & DNA Interactive Quiz")
    quiz = [
        ("Which four letters represent DNA bases?", ["A, T, G, C", "A, U, G, C", "A, B, C, D"], 0, "DNA uses A, T, G and C."),
        ("What does GC content measure?", ["G and C percentage", "DNA length", "Number of proteins"], 0, "It is the percentage of G and C bases."),
        ("Which DNA base pairs with A?", ["G", "T", "C"], 1, "A pairs with T in DNA."),
        ("What is a codon?", ["A group of three bases", "A whole chromosome", "A protein"], 0, "A codon is a triplet of nucleotides."),
        ("Does sequence similarity prove biological activity?", ["Yes", "No"], 1, "A string comparison alone cannot prove biological activity."),
    ]
    quiz_answers = []
    for i, (question, options, correct, explanation) in enumerate(quiz):
        ans = st.radio(question, options, key=f"quiz_{i}")
        quiz_answers.append(options.index(ans))
    if st.button("Submit quiz"):
        score = sum(a == q[2] for a, q in zip(quiz_answers, quiz))
        st.metric("Your score", f"{score}/{len(quiz)}")
        for i, q in enumerate(quiz):
            st.write(("✅" if quiz_answers[i] == q[2] else "❌") + " " + q[3])

elif page == "Results & Reports":
    st.subheader("Results & Reports")
    st.info("Create a summary report for an invented DNA sequence. This report contains descriptive sequence statistics only.")
    seq = clean_sequence(st.text_area("Synthetic DNA sequence for report", "ATGCGCATTA"))
    if validate_dna(seq):
        st.error(validate_dna(seq))
    else:
        report_lines = [
            "GUIDEWISE — EDUCATIONAL DNA SEQUENCE REPORT",
            "Scope: synthetic sequence description only",
            f"Sequence: {seq}",
            f"Length: {len(seq)}",
            f"A: {seq.count('A')} | T: {seq.count('T')} | G: {seq.count('G')} | C: {seq.count('C')}",
            f"GC content: {gc_percentage(seq):.2f}%",
            "This report is not a medical or biological activity prediction.",
        ]
        report_text = "\n".join(report_lines)
        st.text_area("Report preview", report_text, height=180)
        st.download_button(
            "Download TXT report",
            report_text,
            file_name="GUIDEWISE_DNA_report.txt",
            mime="text/plain",
        )
        csv_text = "Nucleotide,Count,Percentage\n" + "\n".join(
            f"{b},{seq.count(b)},{seq.count(b) / len(seq) * 100:.2f}" for b in "ATGC"
        )
        st.download_button(
            "Download composition CSV",
            csv_text,
            file_name="GUIDEWISE_composition.csv",
            mime="text/csv",
        )

elif page == "About & Limitations":
    st.subheader("About GUIDEWISE")
    st.write(
        "GUIDEWISE is an educational software prototype that demonstrates selected "
        "ideas from CRISPR guide matching using synthetic sequences."
    )
    st.markdown("### What the simulator does")
    st.write(
        "- Validates synthetic DNA and guide RNA alphabets.\n"
        "- Compares guide RNA against both orientations of an entered DNA string.\n"
        "- Checks a selected simple PAM pattern separately.\n"
        "- Reports mismatch counts and guide positions.\n"
        "- Compares before/after strings for substitutions, insertions and deletions.\n"
        "- Calculates GC percentage and explains selected biological concepts."
    )
    st.markdown("### Important limitations")
    st.warning(
        "This is not a laboratory protocol, medical application, genome editor, "
        "clinical analysis tool, or experimentally validated predictor."
    )
    st.write(
        "- All examples should be synthetic/artificial strings.\n"
        "- A sequence match does not prove that editing will occur.\n"
        "- A PAM text match does not prove biological activity.\n"
        "- Mismatch count alone cannot establish off-target activity.\n"
        "- The chromatin section is conceptual only.\n"
        "- The mutation checker reports string differences; repeated sequences can make alignment ambiguous.\n"
        "- The hypothetical editing view does not alter DNA or predict a real edit."
    )
    st.markdown("### Suggested hackathon demonstration")
    st.write(
        "1. Open a sample dataset.\n"
        "2. Show a candidate region and its mismatch positions.\n"
        "3. Change the PAM selection and explain that PAM is a separate check.\n"
        "4. Open Mutation Checker and demonstrate a substitution, insertion and deletion.\n"
        "5. Finish with the limitations page to explain the educational scope."
    )


# ---------------------------------------------------------------------
# 12. FOOTER
# ---------------------------------------------------------------------

st.markdown("---")
st.caption("GUIDEWISE • Synthetic-sequence learning prototype • No real editing or biological activity prediction")
