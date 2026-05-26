import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIRI Dashboard — UK Debt Management",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #1F3864; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stSlider label { color: white !important; }
    .main-title { font-size: 28px; font-weight: 700; color: #1F3864; margin-bottom: 4px; }
    .sub-title { font-size: 14px; color: #718096; margin-bottom: 20px; }
    .kpi-box { background: white; border-radius: 10px; padding: 16px; text-align: center;
               border: 1px solid #E2E8F0; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    .kpi-value { font-size: 36px; font-weight: 700; }
    .kpi-label { font-size: 11px; color: #718096; text-transform: uppercase; letter-spacing: 0.5px; }
    .kpi-band { font-size: 12px; font-weight: 700; margin-top: 4px; }
    .section-header { font-size: 16px; font-weight: 700; color: #1F3864;
                      border-left: 4px solid #1F3864; padding-left: 10px; margin: 16px 0 10px 0; }
    .info-box { background: #EEF4FF; border-radius: 8px; padding: 12px;
                border-left: 4px solid #2E5596; font-size: 13px; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ── DATA ──────────────────────────────────────────────────────────────────────
INDICATORS = [
    {"code":"D1-01","name":"data_quality",            "dim":1,"dim_name":"Data Infrastructure",       "default":3,"reg":"GDPR Art.5(1)(d); FCA SYSC 13"},
    {"code":"D1-02","name":"data_governance",          "dim":1,"dim_name":"Data Infrastructure",       "default":3,"reg":"GDPR Art.5(2); FCA DP5/22"},
    {"code":"D1-03","name":"data_integration",         "dim":1,"dim_name":"Data Infrastructure",       "default":3,"reg":"GDPR Art.5(2); FCA Consumer Duty"},
    {"code":"D2-01","name":"system_capability",        "dim":2,"dim_name":"Technological Maturity",    "default":3,"reg":"FCA DP5/22; OECD AI Principle 1.4"},
    {"code":"D2-02","name":"ai_tooling_MLOps",         "dim":2,"dim_name":"Technological Maturity",    "default":3,"reg":"FCA DP5/22; FCA PS21/3"},
    {"code":"D2-03","name":"infrastructure_resilience","dim":2,"dim_name":"Technological Maturity",    "default":3,"reg":"FCA PS21/3 Operational Resilience"},
    {"code":"D3-01","name":"fca_alignment",            "dim":3,"dim_name":"Regulatory Compliance",     "default":3,"reg":"FCA Consumer Duty PS22/9"},
    {"code":"D3-02","name":"consumer_duty",            "dim":3,"dim_name":"Regulatory Compliance",     "default":3,"reg":"FCA Consumer Duty PS22/9 all outcomes"},
    {"code":"D3-03","name":"audit_trail",              "dim":3,"dim_name":"Regulatory Compliance",     "default":3,"reg":"FCA SM&CR; GDPR Art.22"},
    {"code":"D4-01","name":"talent_readiness",         "dim":4,"dim_name":"Organisational Capability", "default":3,"reg":"FCA Consumer Duty competence"},
    {"code":"D4-02","name":"change_management",        "dim":4,"dim_name":"Organisational Capability", "default":3,"reg":"FCA Consumer Duty PS22/9; ISO 42001"},
    {"code":"D4-03","name":"leadership_commitment",    "dim":4,"dim_name":"Organisational Capability", "default":3,"reg":"FCA SM&CR; Consumer Duty PS22/9"},
    {"code":"D5-01","name":"bias_mitigation",          "dim":5,"dim_name":"Ethical Governance",        "default":2,"reg":"FCA Consumer Duty; Equality Act 2010"},
    {"code":"D5-02","name":"explainability",           "dim":5,"dim_name":"Ethical Governance",        "default":2,"reg":"GDPR Art.22; FCA Consumer Duty"},
    {"code":"D5-03","name":"accountability_structures","dim":5,"dim_name":"Ethical Governance",        "default":3,"reg":"FCA Consumer Duty; SM&CR"},
]

DIMENSIONS = {
    1:{"name":"D1: Data Infrastructure",       "weight":0.20,"color":"#1F4E79","light":"#DEEAF1"},
    2:{"name":"D2: Technological Maturity",    "weight":0.20,"color":"#375623","light":"#E2EFDA"},
    3:{"name":"D3: Regulatory Compliance",     "weight":0.25,"color":"#7F3F00","light":"#FFF2CC"},
    4:{"name":"D4: Organisational Capability", "weight":0.15,"color":"#3F1F5F","light":"#EDE7F6"},
    5:{"name":"D5: Ethical Governance",        "weight":0.20,"color":"#4A0000","light":"#FDDCDC"},
}

EVIDENCE = {1:"Not Implemented",2:"Initial Stage",3:"Partially Implemented",
            4:"Substantially Implemented",5:"Fully Evidenced"}

def get_band(score):
    if score >= 76: return "ADVANCED",   "#1F4E79", "#EEF4FF"
    if score >= 51: return "ESTABLISHED","#2E7D32", "#E8F5E9"
    if score >= 26: return "DEVELOPING", "#92600A", "#FFFBEB"
    return               "NASCENT",     "#C62828", "#FFEBEE"

def get_score_color(score):
    if score >= 4: return "#2E7D32"
    if score >= 3: return "#FFC000"
    return "#C62828"

# ── SIDEBAR — SCORE INPUT ─────────────────────────────────────────────────────
st.sidebar.markdown("## 🏦 AIRI Assessment")
st.sidebar.markdown("**Enter indicator scores (1–5)**")
st.sidebar.markdown("---")

institution = st.sidebar.text_input("Institution Name", "Sample UK Debt Firm")
st.sidebar.markdown("---")

scores = {}
for d_id, d_info in DIMENSIONS.items():
    st.sidebar.markdown(f"**{d_info['name']}**")
    st.sidebar.markdown(f"*Weight: {int(d_info['weight']*100)}%*")
    for ind in [i for i in INDICATORS if i["dim"] == d_id]:
        scores[ind["code"]] = st.sidebar.slider(
            f"{ind['code']}: {ind['name']}",
            min_value=1, max_value=5,
            value=ind["default"],
            help=f"1=Not Implemented, 5=Fully Evidenced | {ind['reg']}"
        )
    st.sidebar.markdown("---")

# ── CALCULATE SCORES ──────────────────────────────────────────────────────────
dim_scores = {}
for d_id, d_info in DIMENSIONS.items():
    dim_inds = [i for i in INDICATORS if i["dim"] == d_id]
    avg = sum(scores[i["code"]] for i in dim_inds) / len(dim_inds)
    dim_scores[d_id] = round(avg / 5 * 100, 1)

composite = round(sum(
    dim_scores[d_id] * DIMENSIONS[d_id]["weight"]
    for d_id in DIMENSIONS
), 1)

band_label, band_color, band_bg = get_band(composite)

# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🏦 AI Readiness Index (AIRI) Dashboard</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">UK Debt Management — Organisational AI Governance Assessment | Institution: <strong>{institution}</strong></div>', unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "📐 Dimensions",
    "🔍 Indicators",
    "⚖️ FCA Compliance",
    "⚙️ Sensitivity Analysis"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Composite AIRI Score</div>
            <div class="kpi-value" style="color:{band_color}">{composite}</div>
            <div class="kpi-band" style="color:{band_color};background:{band_bg};padding:3px 10px;border-radius:20px;display:inline-block">{band_label}</div>
            <div class="kpi-label" style="margin-top:4px">out of 100</div>
        </div>""", unsafe_allow_html=True)

    weakest_id = min(dim_scores, key=dim_scores.get)
    strongest_id = max(dim_scores, key=dim_scores.get)

    with col2:
        wname = DIMENSIONS[weakest_id]["name"].split(":")[1].strip()
        wcol  = DIMENSIONS[weakest_id]["color"]
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Weakest Dimension</div>
            <div class="kpi-value" style="color:{wcol};font-size:22px;margin-top:4px">{wname}</div>
            <div class="kpi-band" style="color:{wcol}">{dim_scores[weakest_id]} / 100</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        sname = DIMENSIONS[strongest_id]["name"].split(":")[1].strip()
        scol  = DIMENSIONS[strongest_id]["color"]
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Strongest Dimension</div>
            <div class="kpi-value" style="color:{scol};font-size:22px;margin-top:4px">{sname}</div>
            <div class="kpi-band" style="color:{scol}">{dim_scores[strongest_id]} / 100</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        full_count = sum(1 for v in scores.values() if v == 5)
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Fully Evidenced Indicators</div>
            <div class="kpi-value" style="color:#2E7D32">{full_count}</div>
            <div class="kpi-band" style="color:#2E7D32">of 15 indicators</div>
            <div class="kpi-label" style="margin-top:4px">Scored 5 / 5</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    col_left, col_right = st.columns([1.2, 1])

    # Dimension bar chart
    with col_left:
        st.markdown('<div class="section-header">Dimension Scores</div>', unsafe_allow_html=True)
        df_dims = pd.DataFrame([{
            "Dimension": DIMENSIONS[d]["name"],
            "Score": dim_scores[d],
            "Weight": f"{int(DIMENSIONS[d]['weight']*100)}%",
            "Color":  DIMENSIONS[d]["color"]
        } for d in DIMENSIONS])

        fig_bar = go.Figure()
        for _, row in df_dims.iterrows():
            fig_bar.add_trace(go.Bar(
                x=[row["Score"]], y=[row["Dimension"]],
                orientation="h",
                marker_color=row["Color"],
                text=f"{row['Score']} (Wt: {row['Weight']})",
                textposition="outside",
                showlegend=False,
                hovertemplate=f"<b>{row['Dimension']}</b><br>Score: {row['Score']}/100<br>Weight: {row['Weight']}<extra></extra>"
            ))
        fig_bar.add_vline(x=76, line_dash="dash", line_color="#1F4E79", annotation_text="Advanced (76)")
        fig_bar.add_vline(x=51, line_dash="dash", line_color="#70AD47", annotation_text="Established (51)")
        fig_bar.add_vline(x=26, line_dash="dash", line_color="#FFC000", annotation_text="Developing (26)")
        fig_bar.update_layout(
            xaxis=dict(range=[0,115], title="Score (0–100)"),
            yaxis=dict(autorange="reversed"),
            height=300, margin=dict(l=10,r=10,t=10,b=40),
            plot_bgcolor="white", paper_bgcolor="white"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Radar chart
    with col_right:
        st.markdown('<div class="section-header">Readiness Profile</div>', unsafe_allow_html=True)
        categories = [DIMENSIONS[d]["name"].split(":")[0] for d in DIMENSIONS]
        values     = [dim_scores[d] for d in DIMENSIONS]
        values_closed = values + [values[0]]
        cats_closed   = categories + [categories[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=values_closed, theta=cats_closed,
            fill="toself", fillcolor="rgba(46,85,150,0.2)",
            line=dict(color="#2E5596", width=2.5),
            name="Current State"
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=9)),
                angularaxis=dict(tickfont=dict(size=10))
            ),
            showlegend=False, height=300,
            margin=dict(l=30,r=30,t=30,b=30),
            paper_bgcolor="white"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Priority actions
    st.markdown('<div class="section-header">Top 5 Remediation Priorities</div>', unsafe_allow_html=True)
    sorted_inds = sorted(INDICATORS, key=lambda i: scores[i["code"]])[:5]
    cols = st.columns(5)
    for i, (col, ind) in enumerate(zip(cols, sorted_inds)):
        sc = scores[ind["code"]]
        dcol = DIMENSIONS[ind["dim"]]["color"]
        with col:
            st.markdown(f"""
            <div style="background:{DIMENSIONS[ind['dim']]['light']};border-radius:8px;padding:10px;border-top:4px solid {dcol}">
                <div style="font-size:10px;font-weight:700;color:{dcol}">#{i+1} Priority</div>
                <div style="font-size:12px;font-weight:700;color:{dcol};margin:4px 0">{ind['code']}</div>
                <div style="font-size:10px;color:#444">{ind['name']}</div>
                <div style="font-size:18px;font-weight:700;color:{get_score_color(sc)};margin-top:6px">{sc}/5</div>
                <div style="font-size:9px;color:#666">{EVIDENCE[sc]}</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIMENSIONS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Dimension Score Detail</div>', unsafe_allow_html=True)
    for d_id, d_info in DIMENSIONS.items():
        ds = dim_scores[d_id]
        bl, bc, bb = get_band(ds)
        d_inds = [i for i in INDICATORS if i["dim"] == d_id]
        with st.expander(f"{d_info['name']} — Score: {ds}/100 — {bl}", expanded=(d_id==weakest_id)):
            c1, c2 = st.columns([1,2])
            with c1:
                st.markdown(f"""
                <div style="text-align:center;padding:20px;background:{d_info['light']};border-radius:10px;border:2px solid {d_info['color']}">
                    <div style="font-size:48px;font-weight:700;color:{d_info['color']}">{ds}</div>
                    <div style="font-size:13px;color:{d_info['color']}">out of 100</div>
                    <div style="background:{bc};color:white;padding:4px 12px;border-radius:20px;display:inline-block;font-size:12px;font-weight:700;margin-top:8px">{bl}</div>
                    <div style="font-size:11px;color:#666;margin-top:8px">Weight: {int(d_info['weight']*100)}%</div>
                    <div style="font-size:11px;color:#666">Contribution: {round(ds*d_info['weight'],1)} pts</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                for ind in d_inds:
                    sc = scores[ind["code"]]
                    pct = sc/5*100
                    scol = get_score_color(sc)
                    st.markdown(f"""
                    <div style="background:{d_info['light']};border-radius:8px;padding:10px;margin-bottom:8px;border-left:4px solid {d_info['color']}">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
                            <div>
                                <span style="font-weight:700;color:{d_info['color']};font-size:12px">{ind['code']}: {ind['name']}</span><br>
                                <span style="font-size:10px;color:#666">{ind['reg']}</span>
                            </div>
                            <div style="text-align:right">
                                <div style="font-size:20px;font-weight:700;color:{scol}">{sc}/5</div>
                                <div style="font-size:10px;color:{scol}">{EVIDENCE[sc]}</div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — INDICATORS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">All 15 AIRI Indicators</div>', unsafe_allow_html=True)
    df_ind = pd.DataFrame([{
        "Code":           ind["code"],
        "Indicator":      ind["name"],
        "Dimension":      ind["dim_name"],
        "Score":          scores[ind["code"]],
        "Evidence Status":EVIDENCE[scores[ind["code"]]],
        "Regulatory Basis":ind["reg"]
    } for ind in INDICATORS])

    st.dataframe(
        df_ind,
        use_container_width=True,
        height=500
    )

    st.markdown("")
    st.markdown('<div class="section-header">Score Distribution</div>', unsafe_allow_html=True)
    score_counts = df_ind["Score"].value_counts().sort_index()
    fig_hist = px.bar(
        x=[EVIDENCE[s] for s in score_counts.index],
        y=score_counts.values,
        color=score_counts.index,
        color_continuous_scale=["#FF6B6B","#FFC000","#FFC000","#70AD47","#2E7D32"],
        labels={"x":"Evidence Status","y":"Number of Indicators","color":"Score"},
        text=score_counts.values
    )
    fig_hist.update_traces(textposition="outside")
    fig_hist.update_layout(height=300, showlegend=False,
                           plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10,b=40))
    st.plotly_chart(fig_hist, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — FCA COMPLIANCE
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">FCA Consumer Duty & Regulatory Alignment</div>', unsafe_allow_html=True)

    compliance = [
        {"area":"Consumer Understanding","indicator":"D5-02: explainability",
         "score":scores["D5-02"],"req":"Automated decisions explainable in plain language"},
        {"area":"Consumer Support","indicator":"D5-03: accountability_structures",
         "score":scores["D5-03"],"req":"Human oversight protocols with defined escalation triggers"},
        {"area":"Products & Services","indicator":"D3-02: consumer_duty",
         "score":scores["D3-02"],"req":"Quarterly AI outcome testing across all four Consumer Duty outcomes"},
        {"area":"Price & Value","indicator":"D1-01: data_quality",
         "score":scores["D1-01"],"req":"AI pricing model data quality governed and audited"},
        {"area":"FCA Alignment","indicator":"D3-01: fca_alignment",
         "score":scores["D3-01"],"req":"Formal assessment against FCA Consumer Duty and AI Update 2024"},
        {"area":"SM&CR Accountability","indicator":"D4-03: leadership_commitment",
         "score":scores["D4-03"],"req":"Named SMF holder with AI governance in statement of responsibilities"},
        {"area":"Bias & Fairness","indicator":"D5-01: bias_mitigation",
         "score":scores["D5-01"],"req":"Quarterly algorithmic fairness testing including protected characteristics"},
        {"area":"Audit Trail","indicator":"D3-03: audit_trail",
         "score":scores["D3-03"],"req":"AI governance audit trail available for FCA regulatory inspection"},
        {"area":"Data Governance","indicator":"D1-02: data_governance",
         "score":scores["D1-02"],"req":"Board-approved data governance framework with named data owners"},
    ]

    cols = st.columns(3)
    for i, item in enumerate(compliance):
        sc = item["score"]
        if sc >= 4:   bg,bc,status = "#F0FFF4","#2E7D32","✅ Compliant"
        elif sc >= 3: bg,bc,status = "#FFFBEB","#92600A","⚠️ Partially Compliant"
        else:         bg,bc,status = "#FFF5F5","#C62828","❌ At Risk"
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:{bg};border-radius:8px;padding:12px;margin-bottom:10px;border-left:4px solid {bc}">
                <div style="font-size:11px;font-weight:700;color:{bc}">{item['area']}</div>
                <div style="font-size:10px;color:#444;margin:3px 0">{item['indicator']} — Score: {sc}/5</div>
                <div style="font-size:9px;color:#666;margin-bottom:4px">{item['req']}</div>
                <div style="font-size:11px;font-weight:700;color:{bc}">{status}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="info-box">📋 <strong>Note:</strong> Compliance mapping is based on FCA Consumer Duty PS22/9 (July 2023), SM&CR accountability requirements, and GDPR/Data Protection Act 2018. All indicators should be reviewed against the Scoring Guide for full evidence requirements.</div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Sensitivity Analysis — Adjust Dimension Weights</div>', unsafe_allow_html=True)
    st.markdown("Drag the sliders below to test how alternative weighting assumptions affect the composite AIRI score. Weights must sum to 100%.")

    col_s, col_r = st.columns([1, 1])
    with col_s:
        w1 = st.slider("D1: Data Infrastructure (%)",       0, 50, 20, 5)
        w2 = st.slider("D2: Technological Maturity (%)",    0, 50, 20, 5)
        w3 = st.slider("D3: Regulatory Compliance (%)",     0, 50, 25, 5)
        w4 = st.slider("D4: Organisational Capability (%)", 0, 50, 15, 5)
        w5 = st.slider("D5: Ethical Governance (%)",        0, 50, 20, 5)
        total = w1+w2+w3+w4+w5

        if total == 100:
            st.success(f"✅ Weight total: {total}% — Valid configuration")
        else:
            st.error(f"⚠️ Weight total: {total}% — Must equal 100%")

    with col_r:
        if total == 100:
            adj = round(
                dim_scores[1]*(w1/100) + dim_scores[2]*(w2/100) +
                dim_scores[3]*(w3/100) + dim_scores[4]*(w4/100) +
                dim_scores[5]*(w5/100), 1)
            adj_band, adj_bc, adj_bb = get_band(adj)
            diff = round(adj - composite, 1)
            diff_str = f"+{diff}" if diff > 0 else str(diff)

            st.markdown(f"""
            <div style="background:#1F3864;border-radius:12px;padding:24px;text-align:center;color:white;margin-bottom:16px">
                <div style="font-size:14px;opacity:0.8">Adjusted Composite Score</div>
                <div style="font-size:52px;font-weight:700">{adj}</div>
                <div style="font-size:14px;background:rgba(255,255,255,0.2);padding:4px 16px;border-radius:20px;display:inline-block;margin-top:8px">{adj_band}</div>
                <div style="font-size:13px;margin-top:10px;opacity:0.8">vs Default Score: {composite} &nbsp;|&nbsp; Difference: <strong>{diff_str}</strong></div>
            </div>""", unsafe_allow_html=True)

            # Contribution bars
            weights  = [w1,w2,w3,w4,w5]
            contribs = [round(dim_scores[i+1]*(weights[i]/100),1) for i in range(5)]
            fig_sens = go.Figure()
            for i, (d_id, d_info) in enumerate(DIMENSIONS.items()):
                fig_sens.add_trace(go.Bar(
                    name=d_info["name"],
                    x=[d_info["name"].split(":")[0]],
                    y=[contribs[i]],
                    marker_color=d_info["color"],
                    text=f"{contribs[i]}",
                    textposition="outside"
                ))
            fig_sens.update_layout(
                title="Weighted Contribution per Dimension",
                yaxis_title="Score Contribution",
                height=280, showlegend=False,
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=40,b=20)
            )
            st.plotly_chart(fig_sens, use_container_width=True)
        else:
            st.info("Adjust weights to sum to 100% to see the adjusted score.")

    st.markdown("""
    <div class="info-box">
    📊 <strong>Why sensitivity analysis matters:</strong> This feature was rated the highest Must have item in the expert validation survey (57.9% Must have, 93.4% Must have or Should have). It allows institutions to test how governance priorities — for example, placing greater weight on Regulatory Compliance for a firm under FCA scrutiny — affect their overall readiness score.
    </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:11px;color:#718096">
AIRI — AI Readiness Index for UK Debt Management &nbsp;|&nbsp;
Bournemouth University MSc Information Technology &nbsp;|&nbsp;
COMP7024 Individual Masters Project 2025/26 &nbsp;|&nbsp;
Validated by expert survey (n=121) &nbsp;|&nbsp;
<strong>Sedara Aanuoluwapo Endurance</strong>
</div>
""", unsafe_allow_html=True)
