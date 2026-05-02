"""
Call Center AI — Streamlit UI
==============================

LEARNING: STREAMLIT'S EXECUTION MODEL (read this first)
─────────────────────────────────────────────────────────
Streamlit is fundamentally different from React/Vue. There is no event system,
no virtual DOM, no component lifecycle. Instead:

  1. Every user interaction (button click, file upload, slider move) causes
     Streamlit to RE-RUN THIS ENTIRE SCRIPT from top to bottom.
  2. Regular Python variables (x = 5) are reset on every rerun.
  3. st.session_state is the ONLY place to store data that survives reruns.

Think of each rerun as a "frame" in a game loop — the whole world is redrawn
from scratch, but state is preserved in session_state between frames.

This simplicity is the reason Streamlit apps are so fast to build: you write
plain Python top-to-bottom and Streamlit handles the reactivity automatically.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# LEARNING: Streamlit adds the directory containing this script (ui/) to
# sys.path, not the project root. That means `import pipeline` fails because
# Python looks in ui/ and can't find it.
# We explicitly insert the project root so all project modules are importable
# regardless of where `streamlit run` is invoked from.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import plotly.graph_objects as go
import streamlit as st

# ── Page configuration ────────────────────────────────────────────────────────
# LEARNING: st.set_page_config() MUST be the very first Streamlit call.
# Putting any other st.* call before it raises a StreamlitAPIException.
# layout="wide" removes the default narrow center column and uses full width.
st.set_page_config(
    page_title="Call Center AI",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state initialization ──────────────────────────────────────────────
# LEARNING: Always initialise every session_state key you plan to use at the
# TOP of the script, before any widget reads or writes it.
#
# Why? On the very first run, session_state is empty. If a widget tries to
# read a key that doesn't exist yet, Python raises a KeyError.
# The `if key not in st.session_state` guard means we only set the default
# once — subsequent reruns skip it and keep whatever value is already there.
#
# Analogy: this is the __init__ of your UI's state class.
if "result" not in st.session_state:
    st.session_state.result = None          # Final CallRecord dict from pipeline

if "pipeline_log" not in st.session_state:
    st.session_state.pipeline_log = []      # Per-agent status messages

if "cache_hit" not in st.session_state:
    st.session_state.cache_hit = False      # Whether semantic cache was used


# ── Expensive resource caching ────────────────────────────────────────────────
# LEARNING: @st.cache_resource
#
# build_graph() does a lot of work:
#   - Instantiates 6 agent objects
#   - Each agent creates an OpenAI client
#   - SummarizationAgent loads the sentence-transformer model (~90MB)
#   - LangGraph compiles the StateGraph
#
# Without caching, ALL of this would happen on every single rerun (every click).
# @st.cache_resource runs the function ONCE and reuses the return value forever.
#
# Use @st.cache_resource for:  connections, compiled graphs, ML models
# Use @st.cache_data for:      DataFrames, API responses, serialisable data
@st.cache_resource(show_spinner="⚙️ Loading pipeline (first run only)…")
def load_graph():
    from pipeline.graph import build_graph
    return build_graph()


# ── Helper: run pipeline and return plain dict ────────────────────────────────
def run_pipeline(input_data) -> dict:
    """
    Invokes the LangGraph pipeline and returns the result as a plain dict.

    LEARNING: We convert the Pydantic CallRecord to a plain dict with
    .model_dump(mode="json") before storing it in session_state.

    Why? session_state serialises values to JSON between reruns.
    Pydantic models contain Enum fields (InputType, CallStatus) which are
    not JSON-serialisable by default. mode="json" converts them to strings.
    """
    graph = load_graph()
    state = {"record": input_data}
    final_state = graph.invoke(state)
    record = final_state["record"]
    if hasattr(record, "model_dump"):
        return record.model_dump(mode="json")
    return record


# ── Helper: load sample data ──────────────────────────────────────────────────
# __file__ is ui/streamlit_app.py → .parent is ui/ → .parent is project root
SAMPLE_PATH = Path(__file__).parent.parent / "data" / "sample_result.json"

def load_sample() -> dict:
    with open(SAMPLE_PATH) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("📞 Call Center AI Assistant")
st.caption(
    "Upload a call transcript or audio file → get a structured summary, "
    "key points, action items, and a QA quality score."
)
st.divider()


# ═════════════════════════════════════════════════════════════════════════════
# TABS
# ─────────────────────────────────────────────────────────────────────────────
# LEARNING: st.tabs() creates a horizontal tab bar. All three tab bodies are
# rendered on every rerun — Streamlit does NOT lazy-load inactive tabs.
# This means: never put an expensive computation directly inside a tab body.
# Always guard with `if st.session_state.result is not None`.
# ═════════════════════════════════════════════════════════════════════════════
tab_upload, tab_review, tab_analytics, tab_eval = st.tabs([
    "📤  Upload & Process",
    "📋  Review Results",
    "📊  Analytics",
    "🧪  Evaluation",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — UPLOAD & PROCESS
# ═════════════════════════════════════════════════════════════════════════════
with tab_upload:

    st.subheader("Choose your input")

    # LEARNING: st.radio() with horizontal=True renders options in a row.
    # It returns the selected label as a plain string on every rerun.
    # When the user changes the selection, Streamlit reruns the script and
    # the `if` blocks below render the correct input widget automatically —
    # no routing or event handlers needed.
    input_mode = st.radio(
        "Input mode",
        options=["📄 Paste Transcript", "🎙️ Upload Audio", "🧪 Use Sample Data"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.divider()
    input_data = None   # Reset on every rerun; set below based on mode

    # ── Mode 1: paste JSON transcript ─────────────────────────────────────
    if input_mode == "📄 Paste Transcript":

        st.markdown("**Paste the call transcript:**")

        # LEARNING: st.text_area() returns its current content as a string
        # on every rerun. The `key` parameter gives it a stable identity so
        # Streamlit knows which widget's value to preserve in session_state
        # internally — you don't need to read session_state for it yourself.
        transcript_text = st.text_area(
            label="transcript_input",
            placeholder=(
                "Agent: Thank you for calling support, how can I help?\n"
                "Customer: I have a billing issue on my last invoice.\n"
                "Agent: I can look into that for you right away.\n"
                "Customer: The charge on the 15th looks incorrect."
            ),
            height=220,
            label_visibility="collapsed",
        )

        if transcript_text.strip():
            input_data = {"transcript": transcript_text.strip()}
            st.success("✅ Transcript ready to process")

    # ── Mode 2: audio file upload ──────────────────────────────────────────
    elif input_mode == "🎙️ Upload Audio":

        # LEARNING: st.file_uploader() returns an UploadedFile object,
        # which is a file-like object (has .read(), .name, .size).
        #
        # IMPORTANT: Whisper API needs a real file PATH on disk, not a
        # file-like object. We must write it to a NamedTemporaryFile first.
        # We use delete=False so the file survives the `with` block and
        # can be passed to the pipeline. We clean it up manually afterward.
        uploaded = st.file_uploader(
            "Upload audio file",
            type=["mp3", "wav", "m4a"],
            help="Supported: MP3, WAV, M4A. File will be sent to Whisper API for transcription.",
        )

        if uploaded:
            # LEARNING: st.audio() internally reads the file bytes to render
            # the audio player. This moves the file pointer to the end, so a
            # subsequent uploaded.read() would return empty bytes.
            # uploaded.seek(0) resets the pointer back to the start so the
            # pipeline can read the full file content correctly.
            st.audio(uploaded)
            uploaded.seek(0)    # reset pointer after st.audio() consumed it
            file_size_kb = uploaded.size // 1024
            st.success(f"✅ **{uploaded.name}** ready ({file_size_kb} KB)")
            input_data = uploaded   # Handled below in the run block

    # ── Mode 3: sample data ────────────────────────────────────────────────
    elif input_mode == "🧪 Use Sample Data":

        st.info(
            "📌 **No API calls are made.** This loads a pre-processed result "
            "so you can explore the UI without spending API credits. "
            "Perfect for demos and development."
        )

        # LEARNING: st.button() returns True ONLY on the rerun triggered by
        # the click — it goes back to False on the very next rerun.
        # This is why we immediately write to session_state inside the `if`
        # block: the next rerun won't see True again, but session_state persists.
        if st.button("▶️ Load Sample Data", type="primary"):
            st.session_state.result = load_sample()
            st.session_state.pipeline_log = [
                "⏭️  TranscriptionAgent  — skipped (JSON input)",
                "🚀  SummarizationAgent  — loaded from sample",
                "🚀  QAScoringAgent      — loaded from sample",
            ]
            st.session_state.cache_hit = False
            st.success("✅ Sample loaded! Switch to the **Review Results** tab.")

    # ── Run pipeline button ────────────────────────────────────────────────
    st.divider()

    if input_mode != "🧪 Use Sample Data":

        col_btn, col_hint = st.columns([1, 3])

        with col_btn:
            # LEARNING: disabled= greys out the button when no input is ready.
            # This prevents users from clicking before uploading anything.
            # Streamlit re-evaluates `disabled` on every rerun, so it
            # automatically re-enables as soon as input_data is set.
            run_clicked = st.button(
                "🚀 Run Pipeline",
                type="primary",
                disabled=(input_data is None),
                use_container_width=True,
            )

        with col_hint:
            if input_data is None:
                st.caption("⬆️ Provide an input above to enable this button.")

        if run_clicked and input_data is not None:

            # LEARNING: st.status() is a context manager that shows a live
            # collapsible log box while the enclosed code runs.
            #
            #   expanded=True  → box is open so the user can watch progress
            #   status.update() → call at the end to set final state:
            #     state="complete" shows a green checkmark
            #     state="error"    shows a red X
            #
            # Everything you st.write() INSIDE the `with` block appears
            # as a log line inside the expanding status box.
            with st.status("🔄 Pipeline running…", expanded=True) as status:
                try:
                    actual_input = input_data
                    tmp_path = None

                    # Save uploaded audio to a real temp file on disk
                    if hasattr(input_data, "read"):
                        st.write("📁 Saving audio to temporary file…")
                        suffix = Path(input_data.name).suffix
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=suffix
                        ) as tmp:
                            tmp.write(input_data.read())
                            tmp_path = tmp.name
                        actual_input = tmp_path
                        st.write(f"✅ Saved to `{tmp_path}`")

                    st.write("🔍 IntakeAgent — validating and classifying input…")
                    st.write("🔄 Running agents… (transcription → summarization → QA)")

                    # ── THE ACTUAL PIPELINE CALL ──────────────────────────
                    result = run_pipeline(actual_input)
                    # ─────────────────────────────────────────────────────

                    # Clean up temp audio file
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

                    st.write("✅ Pipeline complete!")

                    # Persist result into session_state so other tabs can read it
                    st.session_state.result = result
                    st.session_state.cache_hit = False

                    # Build a human-readable log based on what actually ran
                    log = ["✅  IntakeAgent       — input validated"]
                    if result.get("input_type") == "audio":
                        log.append("✅  TranscriptionAgent — audio transcribed via Whisper")
                    else:
                        log.append("⏭️  TranscriptionAgent — skipped (JSON input)")

                    if result.get("from_cache"):
                        log.append("🚀  SummarizationAgent — CACHE HIT (LLM skipped)")
                    else:
                        log.append("✅  SummarizationAgent — summary generated via LLM")

                    log.append("✅  QAScoringAgent    — quality scores computed")
                    if result.get("error"):
                        log.append(f"⚠️   Error recorded: {result['error']}")

                    st.session_state.pipeline_log = log
                    status.update(label="✅ Pipeline complete!", state="complete")

                except Exception as exc:
                    status.update(label=f"❌ Pipeline failed: {exc}", state="error")
                    st.error(f"**Error:** {exc}")

        # ── Pipeline log from last run ─────────────────────────────────────
        if st.session_state.pipeline_log:
            st.subheader("Last run")
            for line in st.session_state.pipeline_log:
                st.write(line)

    # ── Nudge user to the next tab when results are ready ─────────────────
    if st.session_state.result:
        st.success("✅ Results ready — switch to the **Review Results** tab ➡️")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — REVIEW RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_review:

    result = st.session_state.result   # Read from session_state, not recomputed

    if result is None:
        st.info(
            "No results yet. Go to **Upload & Process** and run the pipeline first."
        )
    else:
        # ── Metadata strip ─────────────────────────────────────────────────
        # LEARNING: st.columns([1, 1, 1, 2]) creates 4 columns with the last
        # one being twice as wide. col.metric() renders a large-number KPI card
        # with an optional delta arrow (green up / red down).
        c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
        c1.metric("Call ID",  (result.get("call_id") or "—")[:12])
        c2.metric("Status",   (result.get("status") or "—").upper())
        duration = result.get("duration_seconds")
        c3.metric("Duration", f"{int(duration)}s" if duration else "—")
        c4.metric("Agent",    result.get("agent_name") or "Unknown")

        if result.get("from_cache"):
            st.success("🚀 **Cache HIT** — summary was served from semantic cache, LLM was not called.")

        st.divider()

        # ── Two-column layout ──────────────────────────────────────────────
        # LEARNING: gap="large" adds horizontal spacing between columns so
        # content doesn't feel cramped. Streamlit supports "small", "medium",
        # "large" as gap values.
        left_col, right_col = st.columns([1, 1], gap="large")

        with left_col:
            st.subheader("📝 Transcript")

            transcript = result.get("raw_transcript") or "No transcript available."

            # LEARNING: A disabled text_area is a read-only scrollable box.
            # It's better than st.write() for long text because it scrolls
            # instead of pushing down all content below it.
            st.text_area(
                label="raw_transcript",
                value=transcript,
                height=400,
                disabled=True,
                label_visibility="collapsed",
            )

        with right_col:
            st.subheader("🧠 Summary")
            st.write(result.get("summary") or "_No summary generated._")

            st.divider()
            st.subheader("📌 Key Points")
            key_points = result.get("key_points") or []
            if key_points:
                for point in key_points:
                    st.markdown(f"- {point}")
            else:
                st.write("_No key points extracted._")

            st.divider()
            st.subheader("✅ Action Items")
            action_items = result.get("action_items") or []
            if action_items:
                for i, item in enumerate(action_items):
                    # LEARNING: Every widget that appears in a loop MUST have
                    # a unique key. Without it, Streamlit can't distinguish
                    # between widgets and raises a DuplicateWidgetID error.
                    # We use f"action_{i}" to make each key unique.
                    st.checkbox(item, value=False, key=f"action_{i}")
            else:
                st.write("_No action items._")

        # ── Error banner ───────────────────────────────────────────────────
        if result.get("error"):
            st.divider()
            st.error(f"⚠️ Pipeline recorded an error: `{result['error']}`")

        # ── Raw output expander ────────────────────────────────────────────
        # LEARNING: st.expander() hides verbose content behind a toggle.
        # Great for debug info — advanced users can open it, others ignore it.
        st.divider()
        with st.expander("🔍 Raw pipeline output (debug)"):
            st.json(result)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════
with tab_analytics:

    result = st.session_state.result

    if result is None:
        st.info("No results yet. Process a call first.")
    else:
        qa = result.get("qa_scores") or {}

        if not qa:
            st.warning("QA scores not available for this result.")
        else:
            DIMENSIONS = ["empathy", "resolution", "tone", "professionalism"]
            ICONS = {
                "empathy": "❤️",
                "resolution": "✅",
                "tone": "🎙️",
                "professionalism": "🏢",
            }

            overall = qa.get("overall_score", 0.0)

            # ── Header KPI ─────────────────────────────────────────────────
            st.subheader("📊 Quality Score Breakdown")

            # LEARNING: delta_color="normal" → positive delta = green,
            # negative = red. "inverse" flips it. "off" removes colour.
            # The delta shows deviation from the midpoint score (3.0),
            # giving the reviewer instant context: good or bad?
            st.metric(
                label="🏆 Overall Score",
                value=f"{overall:.2f} / 5.0",
                delta=f"{overall - 3.0:+.2f} vs midpoint (3.0)",
                delta_color="normal",
            )

            st.divider()

            # ── Individual dimension cards ─────────────────────────────────
            # LEARNING: zip(cols, DIMENSIONS) is a clean pattern to assign
            # one widget per column without writing four identical blocks.
            cols = st.columns(4)
            for col, dim in zip(cols, DIMENSIONS):
                score = qa.get(dim, 0.0)
                with col:
                    st.metric(
                        label=f"{ICONS[dim]} {dim.title()}",
                        value=f"{score:.1f}",
                        delta=f"{score - 3.0:+.1f}",
                    )
                    # LEARNING: st.progress() expects a float between 0.0–1.0.
                    # Scores are 1–5, so we normalise: (score - 1) / 4
                    # Score 1 → 0.0 (empty bar)
                    # Score 5 → 1.0 (full bar)
                    st.progress((score - 1.0) / 4.0)

            st.divider()

            # ── Radar chart ────────────────────────────────────────────────
            # LEARNING: Plotly's Scatterpolar creates a radar (spider) chart.
            # It's excellent for multi-dimension scoring because you can see
            # the "shape" of performance at a glance — a regular pentagon
            # means balanced performance; spikes/dips show strengths/weaknesses.
            #
            # We close the polygon by repeating the first value at the end
            # (r + [r[0]], theta + [theta[0]]).
            st.subheader("🕸️ Performance Radar")

            scores = [qa.get(d, 0.0) for d in DIMENSIONS]
            labels = [f"{ICONS[d]} {d.title()}" for d in DIMENSIONS]

            fig = go.Figure(
                data=go.Scatterpolar(
                    r=scores + [scores[0]],               # close the polygon
                    theta=labels + [labels[0]],
                    fill="toself",
                    fillcolor="rgba(99, 110, 250, 0.25)",
                    line=dict(color="rgb(99, 110, 250)", width=2),
                    name="QA Score",
                ),
                layout=go.Layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 5],
                            tickvals=[1, 2, 3, 4, 5],
                        )
                    ),
                    showlegend=False,
                    margin=dict(t=40, b=40, l=60, r=60),
                    height=380,
                ),
            )

            # LEARNING: use_container_width=True makes the chart fill the
            # column width responsively instead of using a fixed pixel size.
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # ── Bar chart: scores vs benchmark ─────────────────────────────
            # LEARNING: Plotly grouped bar charts are built by adding multiple
            # Bar traces to the same Figure. Each trace is one "group" —
            # here: actual scores vs a 3.0 benchmark line.
            st.subheader("📊 Scores vs Benchmark (3.0)")

            bar_fig = go.Figure(
                data=[
                    go.Bar(
                        name="Actual Score",
                        x=[d.title() for d in DIMENSIONS],
                        y=[qa.get(d, 0.0) for d in DIMENSIONS],
                        marker_color=[
                            "green" if qa.get(d, 0) >= 3.0 else "red"
                            for d in DIMENSIONS
                        ],
                        text=[f"{qa.get(d, 0.0):.1f}" for d in DIMENSIONS],
                        textposition="outside",
                    )
                ],
                layout=go.Layout(
                    yaxis=dict(range=[0, 5.5], title="Score"),
                    xaxis=dict(title="Dimension"),
                    shapes=[
                        # Dashed horizontal line at the benchmark score
                        dict(
                            type="line",
                            y0=3.0, y1=3.0,
                            x0=-0.5, x1=len(DIMENSIONS) - 0.5,
                            line=dict(color="orange", width=2, dash="dash"),
                        )
                    ],
                    annotations=[
                        dict(
                            x=len(DIMENSIONS) - 0.5,
                            y=3.1,
                            text="Benchmark 3.0",
                            showarrow=False,
                            font=dict(color="orange", size=11),
                        )
                    ],
                    height=350,
                    margin=dict(t=30, b=40),
                    showlegend=False,
                ),
            )
            st.plotly_chart(bar_fig, use_container_width=True)

            st.divider()

            # ── Written interpretation ─────────────────────────────────────
            st.subheader("📋 Interpretation")

            if overall >= 4.0:
                st.success(
                    "🟢 **Excellent call.** Agent performed above standard across all "
                    "dimensions. No action required."
                )
            elif overall >= 3.0:
                st.info(
                    "🟡 **Acceptable call.** Performance meets the baseline but has "
                    "room for improvement in at least one dimension."
                )
            else:
                st.error(
                    "🔴 **Below standard.** This call was flagged for escalation. "
                    "Review recommended."
                )

            # Highlight the weakest dimension
            scored = {d: qa[d] for d in DIMENSIONS if d in qa}
            if scored:
                weakest_dim = min(scored, key=scored.get)
                weakest_val = scored[weakest_dim]
                if weakest_val < 4.5:
                    st.warning(
                        f"💡 **Focus area:** `{weakest_dim.title()}` scored "
                        f"{weakest_val:.1f} / 5.0 — the lowest dimension for this call."
                    )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — EVALUATION HARNESS
# ─────────────────────────────────────────────────────────────────────────────
# LEARNING: The evaluation tab runs our offline and LLM-based metrics against
# a pre-annotated dataset (data/eval_dataset.json).
#
# Metrics recap:
#   Token F1   — bag-of-words precision/recall/F1. Fast, no models needed.
#   ROUGE-L    — Longest Common Subsequence overlap. Captures sentence order.
#   BERTScore  — Contextual BERT embeddings. Handles synonyms and paraphrasing.
#   RAGAS      — LLM-judged: faithfulness, answer_relevancy, context metrics.
#
# Why keep offline (F1, ROUGE, BERT) and LLM-based (RAGAS) separate?
#   Offline metrics are deterministic, cheap, and fast — great for CI/CD.
#   RAGAS uses LLM calls — slower and costs money. Run it on-demand.
# ═════════════════════════════════════════════════════════════════════════════
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "eval_results.json"

with tab_eval:
    st.subheader("🧪 Evaluation Harness")
    st.caption(
        "Runs automated metrics on the pre-annotated `data/eval_dataset.json`. "
        "Token F1, ROUGE-L, and BERTScore run offline. "
        "RAGAS requires OpenAI API calls."
    )

    # ── Metric explanation expander ───────────────────────────────────────
    with st.expander("📚 What do these metrics measure?", expanded=False):
        st.markdown("""
| Metric | Type | What it measures | Handles synonyms? |
|---|---|---|---|
| **Token F1** | Lexical | Bag-of-words precision/recall between reference and candidate | ❌ |
| **ROUGE-L** | Lexical | Longest Common Subsequence overlap (respects word order) | ❌ |
| **BERTScore** | Semantic | Cosine similarity of contextual BERT embeddings | ✅ |
| **RAGAS Faithfulness** | LLM-judged | Fraction of answer claims supported by retrieved context | ✅ |
| **RAGAS Answer Relevancy** | LLM-judged | How well the answer addresses the original question | ✅ |
| **RAGAS Context Recall** | LLM-judged | How much of the reference answer is covered by retrieved context | ✅ |
| **RAGAS Context Precision** | LLM-judged | What fraction of retrieved chunks were actually useful | ✅ |

**F1 = 2 × (Precision × Recall) / (Precision + Recall)**
A score of 1.0 is perfect; 0.0 is no overlap.
        """)

    st.divider()

    # ── Run controls ──────────────────────────────────────────────────────
    col_offline, col_ragas, col_spacer = st.columns([1, 1, 2])

    with col_offline:
        run_offline = st.button(
            "▶️ Run Offline Metrics",
            type="primary",
            use_container_width=True,
            help="Runs Token F1, ROUGE-L, BERTScore — no API calls needed",
        )

    with col_ragas:
        run_with_ragas = st.button(
            "🧠 Run + RAGAS",
            use_container_width=True,
            help="Runs all metrics including RAGAS (uses OpenAI API)",
        )

    # ── Execute evaluation ─────────────────────────────────────────────────
    if run_offline or run_with_ragas:
        skip_ragas = not run_with_ragas

        with st.status(
            "🔄 Running evaluation…" if skip_ragas else "🔄 Running evaluation + RAGAS…",
            expanded=True,
        ) as eval_status:
            try:
                from evaluation.run_eval import run_evaluation

                st.write("📂 Loading eval_dataset.json …")
                st.write("📐 Computing Token F1 …")
                st.write("📐 Computing ROUGE-L …")
                st.write("📐 Computing BERTScore (downloads ~250 MB on first run) …")
                if not skip_ragas:
                    st.write("🤖 Running RAGAS (LLM calls in progress) …")

                eval_results = run_evaluation(skip_ragas=skip_ragas)

                # Persist so we can re-display without re-running
                st.session_state["eval_results"] = eval_results
                eval_status.update(label="✅ Evaluation complete!", state="complete")

            except Exception as exc:
                eval_status.update(label=f"❌ Evaluation failed: {exc}", state="error")
                st.error(f"**Error:** {exc}")

    # ── Display stored results ─────────────────────────────────────────────
    # LEARNING: We check session_state first so results survive tab switches.
    # If no session_state entry, fall back to the saved JSON file from last run.
    stored_results = st.session_state.get("eval_results")

    if stored_results is None and EVAL_RESULTS_PATH.exists():
        with open(EVAL_RESULTS_PATH) as f:
            stored_results = json.load(f)
        st.info("📂 Showing results from previous run (`data/eval_results.json`). Click a button above to re-run.")

    if stored_results:
        st.divider()
        st.subheader("📊 Results by Sample")

        # ── Per-sample metric cards ────────────────────────────────────────
        for r in stored_results:
            s = r["scores"]
            tf1 = s.get("token_f1", {})
            rl  = s.get("rouge_l", {})
            bs  = s.get("bertscore", {})
            rag = s.get("ragas", {})

            with st.expander(f"**{r['id']}** — {r['scenario']}", expanded=True):
                m1, m2, m3 = st.columns(3)

                m1.metric(
                    "Token F1",
                    f"{tf1.get('f1', 0):.3f}",
                    help="Bag-of-words F1. ≥ 0.5 is generally good.",
                )
                m2.metric(
                    "ROUGE-L F1",
                    f"{rl.get('f1', 0):.3f}",
                    help="LCS-based overlap. ≥ 0.4 is generally good.",
                )
                bs_f1 = bs.get("f1")
                if bs_f1 is not None:
                    m3.metric(
                        "BERTScore F1",
                        f"{bs_f1:.3f}",
                        help="Semantic similarity. ≥ 0.85 is generally good.",
                    )
                else:
                    m3.metric("BERTScore F1", "N/A", help=bs.get("error", "Unavailable"))
                    m3.caption("⚠️ Model download required")

                if rag and "error" not in rag and any(v is not None for v in rag.values()):
                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Faithfulness",     f"{rag.get('faithfulness', 0):.3f}")
                    r2.metric("Answer Relevancy", f"{rag.get('answer_relevancy', 0):.3f}")
                    r3.metric("Context Recall",   f"{rag.get('context_recall', 0):.3f}")
                    r4.metric("Context Precision",f"{rag.get('context_precision', 0):.3f}")
                elif not rag:
                    st.caption("RAGAS: not run (use **Run + RAGAS** button)")
                elif "error" in rag:
                    st.warning(f"RAGAS error: {rag['error']}")

        # ── Aggregate bar chart ────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Aggregate Comparison — F1 Scores Across Samples")

        labels = [r["id"] for r in stored_results]
        tf1_vals  = [r["scores"].get("token_f1",  {}).get("f1") or 0 for r in stored_results]
        rl_vals   = [r["scores"].get("rouge_l",   {}).get("f1") or 0 for r in stored_results]
        bs_vals   = [r["scores"].get("bertscore", {}).get("f1") or 0 for r in stored_results]

        # LEARNING: Grouped bar chart — three traces (one per metric), same x-axis.
        # barmode="group" places bars side-by-side instead of stacking.
        agg_fig = go.Figure(
            data=[
                go.Bar(name="Token F1",   x=labels, y=tf1_vals, marker_color="#636EFA"),
                go.Bar(name="ROUGE-L F1", x=labels, y=rl_vals,  marker_color="#EF553B"),
                go.Bar(name="BERTScore",  x=labels, y=bs_vals,  marker_color="#00CC96"),
            ],
            layout=go.Layout(
                barmode="group",
                yaxis=dict(range=[0, 1.0], title="F1 Score"),
                xaxis=dict(title="Eval Sample"),
                height=380,
                margin=dict(t=30, b=50),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            ),
        )
        st.plotly_chart(agg_fig, use_container_width=True)

        # ── Summary averages table ─────────────────────────────────────────
        st.divider()
        st.subheader("📋 Average Scores")

        def _avg(vals):
            valid = [v for v in vals if v is not None]
            return round(sum(valid) / len(valid), 4) if valid else "N/A"

        def _safe(d, key):
            v = d.get(key)
            return v if v is not None else None

        avg_data = {
            "Metric": ["Token F1", "ROUGE-L F1", "BERTScore F1"],
            "Avg Precision": [
                _avg([_safe(r["scores"].get("token_f1",  {}), "precision") for r in stored_results]),
                _avg([_safe(r["scores"].get("rouge_l",   {}), "precision") for r in stored_results]),
                _avg([_safe(r["scores"].get("bertscore", {}), "precision") for r in stored_results]),
            ],
            "Avg Recall": [
                _avg([_safe(r["scores"].get("token_f1",  {}), "recall") for r in stored_results]),
                _avg([_safe(r["scores"].get("rouge_l",   {}), "recall") for r in stored_results]),
                _avg([_safe(r["scores"].get("bertscore", {}), "recall") for r in stored_results]),
            ],
            "Avg F1": [
                _avg(tf1_vals),
                _avg(rl_vals),
                _avg([v if v > 0 else None for v in bs_vals]),
            ],
        }

        # LEARNING: st.table() renders a static HTML table (no sorting/filtering).
        # For interactive tables use st.dataframe(). We use st.table() here
        # because the data is tiny and we don't need interactivity.
        import pandas as pd
        st.table(pd.DataFrame(avg_data).set_index("Metric"))

    else:
        st.info("Click **▶️ Run Offline Metrics** to evaluate the annotated dataset.")
