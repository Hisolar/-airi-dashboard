"""
AIRI - AI Readiness Index for UK Debt Management
Bournemouth University | MSc Information Technology | COMP7024 | 2025/26
Sedara Aanuoluwapo Endurance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json, os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIRI - AI Readiness Index",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* White background everywhere */
    .stApp, .main { background-color: #ffffff; }
    .main .block-container { padding-top: 1.5rem; background-color: #ffffff; }

    /* All text black */
    body, p, span, div, li, td, th,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stText"] { color: #111111 !important; font-family: Arial, sans-serif; }

    /* Headings */
    h1 { color: #111111 !important; font-size: 1.9rem; font-weight: 700; }
    h2 { color: #111111 !important; font-size: 1.3rem; font-weight: 700;
         border-bottom: 2px solid #007700; padding-bottom: 5px; margin-top: 1.5rem; }
    h3 { color: #111111 !important; font-size: 1.05rem; font-weight: 700; }

    /* Form labels */
    label, .stSlider label,
    .stSelectbox label,
    [data-testid="stWidgetLabel"] { color: #111111 !important; font-weight: 500; }

    /* Buttons — green with white text */
    .stButton > button {
        background-color: #007700 !important;
        color: #ffffff !important;
        font-weight: 700;
        border-radius: 4px;
        border: none;
        padding: 8px 20px;
    }
    .stButton > button:hover {
        background-color: #005500 !important;
        color: #ffffff !important;
    }

    /* Sidebar — white background, black text */
    [data-testid="stSidebar"] {
        background-color: #f8f8f8;
        border-right: 2px solid #dddddd;
    }
    [data-testid="stSidebar"] * { color: #111111 !important; }
    [data-testid="stSidebar"] .stButton > button {
        background-color: #007700 !important;
        color: #ffffff !important;
        font-weight: 600;
        margin-bottom: 4px;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #005500 !important;
    }

    /* Expander */
    .streamlit-expanderHeader { color: #111111 !important; font-weight: 600; }
    .streamlit-expanderHeader p { color: #111111 !important; }
    .streamlit-expanderContent p { color: #111111 !important; }
    [data-testid="stExpanderDetails"] p { color: #111111 !important; }

    /* Metrics */
    [data-testid="stMetricValue"] { color: #007700 !important; font-weight: 700; }
    [data-testid="stMetricLabel"] { color: #111111 !important; }

    /* Tables */
    .stDataFrame td { color: #111111 !important; }
    .stDataFrame th { color: #111111 !important; background-color: #f0f0f0 !important; }

    /* Alert boxes */
    .stAlert p { color: #111111 !important; }
    [data-testid="stNotification"] p { color: #111111 !important; }

    /* Selectbox and slider text */
    .stSelectbox div { color: #111111 !important; }
    .stSlider p { color: #111111 !important; }

    /* Download button */
    .stDownloadButton > button {
        background-color: #007700 !important;
        color: #ffffff !important;
        font-weight: 700;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
GREEN      = "#007700"
DKGREEN    = "#005500"
BLACK      = "#111111"
WHITE      = "#ffffff"
LIGHTGREY  = "#f4f4f4"
GREY       = "#dddddd"
LINEGREEN  = "#007700"

# ── READINESS BANDS ────────────────────────────────────────────────────────────
def get_bands():
    return {
        "Nascent":     {"min":0,  "max":25,  "bg":"#ffeeee","border":"#cc0000","text":"#cc0000",
            "desc":"Fundamental gaps in AI readiness. Early-stage exploration with limited infrastructure, governance, or skills.",
            "recs":["Establish basic data governance policies","Develop initial AI strategy and roadmap",
                    "Conduct AI literacy training for leadership","Review FCA Consumer Duty requirements",
                    "Create ethical AI principles document"]},
        "Developing":  {"min":26, "max":50,  "bg":"#fff8e0","border":"#cc7700","text":"#885500",
            "desc":"Building core capabilities. Partial implementation with identified gaps in technology, governance, or workforce.",
            "recs":["Implement data quality monitoring tools","Establish MLOps pipelines for model deployment",
                    "Expand AI training to operational staff","Develop vulnerability-sensitive customer protocols",
                    "Create model explainability frameworks"]},
        "Established": {"min":51, "max":75,  "bg":"#eeffee","border":"#007700","text":"#005500",
            "desc":"Mature AI practices. Operational AI with robust governance, compliance, and continuous monitoring.",
            "recs":["Optimise AI model performance and drift monitoring","Enhance cross-functional AI governance committees",
                    "Implement advanced bias detection algorithms","Develop peer benchmarking capabilities",
                    "Create automated compliance reporting"]},
        "Advanced":    {"min":76, "max":100, "bg":"#eeeeff","border":"#000088","text":"#000088",
            "desc":"Leading-edge AI readiness. Continuous innovation, proactive governance, and industry-leading practices.",
            "recs":["Pioneer new AI governance standards","Develop AI-driven regulatory horizon scanning",
                    "Create industry collaboration frameworks","Implement real-time ethical AI monitoring",
                    "Establish AI research and innovation labs"]},
    }

def get_conceptual_dims():
    return {
        "Data Infrastructure":       {"code":"D1","desc":"Quality, integration, governance, and security of data assets",
            "inds":["Data quality standards and monitoring","Data integration across systems",
                    "Data governance framework","Data security and privacy controls","Real-time data availability"]},
        "Technological Maturity":    {"code":"D2","desc":"Technical infrastructure and MLOps capabilities",
            "inds":["Cloud infrastructure readiness","MLOps and model deployment pipelines",
                    "API and integration architecture","Computational resources for AI","Monitoring and observability tools"]},
        "Regulatory Compliance":     {"code":"D3","desc":"FCA compliance, data protection, cybersecurity, vulnerability protocols",
            "inds":["FCA Consumer Duty alignment","GDPR and data protection compliance",
                    "Cybersecurity framework","Vulnerable customer protocols","Audit trail and documentation"]},
        "Organisational Capability": {"code":"D4","desc":"Leadership, skills, culture, and resources for AI adoption",
            "inds":["Executive AI leadership and sponsorship","AI literacy and training programs",
                    "Cross-functional AI teams","Change management capability","Budget and resource allocation"]},
        "Ethical Governance":        {"code":"D5","desc":"Bias mitigation, fairness, transparency, and consumer protection",
            "inds":["Algorithmic bias detection and mitigation","Explainability and interpretability frameworks",
                    "Fairness assessment procedures","Consumer protection safeguards","Ethical review board or committee"]},
    }

def get_questions():
    return {
        "Strategy & Governance": [
            {"id":"Q27","text":"Our organisation has a documented AI strategy aligned with business objectives and FCA expectations.","sub":"AI Strategy"},
            {"id":"Q28","text":"Clear decision rights and accountability structures exist for AI system deployment and oversight.","sub":"Decision Rights"},
            {"id":"Q29","text":"We have established AI governance committees with cross-functional representation (IT, Risk, Compliance, Legal).","sub":"Governance Structure"},
            {"id":"Q30","text":"Our AI governance framework explicitly addresses the FCA Consumer Duty requirements for fair outcomes.","sub":"FCA Alignment"},
            {"id":"Q31","text":"We maintain comprehensive audit trails for all AI-driven decisions affecting customers.","sub":"Audit & Accountability"},
            {"id":"Q32","text":"Regular board-level reviews of AI risks, performance, and strategic alignment are conducted.","sub":"Board Oversight"},
            {"id":"Q33","text":"Our organisation has defined AI risk appetite statements integrated into enterprise risk management.","sub":"Risk Appetite"},
            {"id":"Q34","text":"We have established clear escalation procedures for AI incidents and customer complaints.","sub":"Incident Management"},
        ],
        "Data & Technology": [
            {"id":"Q11","text":"Our data infrastructure supports real-time or near-real-time data processing for AI applications.","sub":"Data Infrastructure"},
            {"id":"Q12","text":"Data quality is systematically monitored with defined metrics and remediation procedures.","sub":"Data Quality"},
            {"id":"Q13","text":"We have implemented data lineage tracking to understand data provenance for AI model inputs.","sub":"Data Lineage"},
            {"id":"Q14","text":"Our organisation has cloud-based or scalable on-premise infrastructure for AI model training and deployment.","sub":"Cloud Infrastructure"},
            {"id":"Q15","text":"MLOps practices (version control, CI/CD, model registry) are implemented for AI lifecycle management.","sub":"MLOps"},
            {"id":"Q16","text":"APIs and integration layers enable seamless data flow between operational systems and AI platforms.","sub":"Integration"},
            {"id":"Q17","text":"We have adequate computational resources (GPU/TPU) for training and inference of AI models.","sub":"Compute Resources"},
            {"id":"Q18","text":"Data security controls (encryption, access controls, anonymization) meet financial services standards.","sub":"Data Security"},
            {"id":"Q19","text":"Monitoring and observability tools track AI model performance, drift, and operational health.","sub":"Monitoring"},
            {"id":"Q20","text":"Our data architecture supports integration of structured and unstructured data for AI applications.","sub":"Data Architecture"},
        ],
        "People & Skills": [
            {"id":"Q21","text":"Executive leadership demonstrates visible commitment and sponsorship for AI initiatives.","sub":"Leadership"},
            {"id":"Q22","text":"We have conducted AI literacy assessments and identified skill gaps across the organisation.","sub":"AI Literacy"},
            {"id":"Q23","text":"Role-based AI training programs are available for technical and non-technical staff.","sub":"Training Programs"},
            {"id":"Q24","text":"Cross-functional AI teams (data scientists, engineers, domain experts) are established and resourced.","sub":"Team Structure"},
            {"id":"Q25","text":"Change management processes support AI adoption and address workforce transition concerns.","sub":"Change Management"},
            {"id":"Q26","text":"We have access to external AI expertise (consultants, vendors, academic partnerships) when needed.","sub":"External Expertise"},
        ],
        "Risk & Ethics": [
            {"id":"Q35","text":"We have implemented procedures to detect and mitigate algorithmic bias in AI models.","sub":"Bias Mitigation"},
            {"id":"Q36","text":"AI model decisions can be explained to regulators, customers, and internal stakeholders.","sub":"Explainability"},
            {"id":"Q37","text":"Fairness assessments are conducted regularly to ensure equitable outcomes across customer segments.","sub":"Fairness"},
            {"id":"Q38","text":"Our AI systems incorporate vulnerability-sensitive design for financially distressed customers.","sub":"Vulnerability Sensitivity"},
            {"id":"Q39","text":"Human-in-the-loop protocols ensure meaningful oversight of high-stakes AI decisions.","sub":"Human Oversight"},
            {"id":"Q40","text":"Ethical review processes evaluate AI use cases before deployment.","sub":"Ethical Review"},
            {"id":"Q41","text":"We have documented procedures for handling AI-related customer complaints and remediation.","sub":"Consumer Protection"},
        ],
    }

# ── SCORING ────────────────────────────────────────────────────────────────────
def norm(raw, mn=0, mx=8):
    if mx == mn: return 50.0
    return max(0.0, min(100.0, (raw - mn) / (mx - mn) * 100))

def dim_score(responses, qs):
    vals = [norm(responses[q["id"]]) for q in qs if q["id"] in responses]
    return round(float(np.mean(vals)), 2) if vals else 0.0

def composite(dim_scores):
    return round(float(np.mean(list(dim_scores.values()))), 2)

def classify(score):
    for name, b in get_bands().items():
        if name == "Advanced":
            if b["min"] <= score <= b["max"]: return name
        else:
            if b["min"] <= score < b["max"]: return name
    return "Unknown"

# ── CHARTS ─────────────────────────────────────────────────────────────────────
def ch_radar(ds, title="AIRI Dimension Profile"):
    cats = list(ds.keys()); vals = list(ds.values())
    cc = cats + [cats[0]]; vc = vals + [vals[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vc, theta=cc, fill='toself',
        fillcolor='rgba(0,119,0,0.12)', line=dict(color=GREEN, width=2.5), name='Score'))
    for v, lbl in [(25,'Nascent'),(50,'Developing'),(75,'Established')]:
        fig.add_trace(go.Scatterpolar(r=[v]*(len(cats)+1), theta=cc, mode='lines',
            line=dict(dash='dot', width=1, color='#999999'), name=lbl, opacity=0.7))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,100], tickvals=[0,25,50,75,100],
                tickfont=dict(size=10, color=BLACK), gridcolor='#cccccc'),
            angularaxis=dict(tickfont=dict(size=11, color=BLACK)),
            bgcolor=WHITE),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5, font=dict(color=BLACK, size=10)),
        title=dict(text=title, font=dict(size=15, color=BLACK), x=0.5),
        paper_bgcolor=WHITE, height=440, margin=dict(t=60, b=90))
    return fig

def ch_bar(ds):
    dims = list(ds.keys()); scores = list(ds.values())
    colors = ['#ffcccc' if s<25 else '#ffe8a0' if s<50 else '#bbffbb' if s<75 else '#aaccff' for s in scores]
    fig = go.Figure(go.Bar(x=scores, y=dims, orientation='h',
        marker=dict(color=colors, line=dict(color='#333333', width=0.8)),
        text=[f'{s:.1f}' for s in scores], textposition='auto',
        textfont=dict(size=12, color=BLACK)))
    for v, lbl in [(25,'Nascent'),(50,'Developing'),(75,'Established')]:
        fig.add_vline(x=v, line_dash="dash", line_color='#666666', line_width=1.2,
                      annotation_text=lbl, annotation_font=dict(color=BLACK, size=10))
    fig.update_layout(
        xaxis=dict(range=[0,110], title=dict(text="Score (0–100)", font=dict(color=BLACK)),
                   tickfont=dict(color=BLACK), gridcolor='#eeeeee'),
        yaxis=dict(tickfont=dict(color=BLACK, size=11)),
        title=dict(text="Dimension Score Breakdown", font=dict(size=15, color=BLACK), x=0.5),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        height=360, margin=dict(l=170, r=60, t=50, b=50))
    return fig

def ch_gauge(score):
    band = classify(score)
    binfo = get_bands()[band]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={'suffix':"/100", 'font':{'size':42, 'color':BLACK}},
        title={'text':"AIRI Composite Score", 'font':{'size':15, 'color':BLACK}},
        gauge={'axis':{'range':[0,100], 'tickfont':{'color':BLACK, 'size':10}},
               'bar':{'color':GREEN, 'thickness':0.65},
               'bgcolor':WHITE, 'borderwidth':1, 'bordercolor':GREY,
               'steps':[{'range':[0,25],'color':'#ffeeee'},{'range':[25,50],'color':'#fff8e0'},
                        {'range':[50,75],'color':'#eeffee'},{'range':[75,100],'color':'#eeeeff'}],
               'threshold':{'line':{'color':BLACK,'width':3},'thickness':0.8,'value':score}}))
    fig.update_layout(height=340, paper_bgcolor=WHITE, margin=dict(t=60,b=20))
    return fig

def ch_impact(ds):
    dims = list(ds.keys()); scores = list(ds.values())
    mean = float(np.mean(scores))
    devs = [s - mean for s in scores]
    colors = ['#cc0000' if d < 0 else GREEN for d in devs]
    fig = go.Figure(go.Bar(y=dims, x=devs, orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=0.5)),
        text=[f'{d:+.1f}' for d in devs], textposition='outside',
        textfont=dict(size=11, color=BLACK)))
    fig.add_vline(x=0, line_color=BLACK, line_width=1.5)
    fig.update_layout(
        title=dict(text="Dimension Impact — Deviation from Mean",
                   font=dict(size=15, color=BLACK), x=0.5),
        xaxis=dict(title=dict(text="Impact", font=dict(color=BLACK)),
                   tickfont=dict(color=BLACK),
                   zeroline=True, zerolinecolor=BLACK, zerolinewidth=1.5,
                   gridcolor='#eeeeee'),
        yaxis=dict(tickfont=dict(color=BLACK, size=11)),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        height=300, margin=dict(l=170, r=80, t=50, b=40),
        showlegend=False)
    return fig

def ch_pca(df):
    if len(df) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 3 assessments for PCA",
                           showarrow=False, font=dict(size=13, color=BLACK))
        fig.update_layout(height=360, paper_bgcolor=WHITE,
                          xaxis=dict(visible=False), yaxis=dict(visible=False))
        return fig
    dim_cols = [c for c in ['Strategy & Governance','Data & Technology','People & Skills','Risk & Ethics'] if c in df.columns]
    if not dim_cols:
        dim_cols = [c for c in df.columns if c not in ['AIRI_composite','AIRI_band']]
    X = df[dim_cols].values
    X_s = MinMaxScaler().fit_transform(X)
    pca = PCA(n_components=2)
    X_p = pca.fit_transform(X_s)
    n_c = min(4, len(df))
    clusters = KMeans(n_clusters=n_c, random_state=42, n_init='auto').fit_predict(X_s)
    pdf = pd.DataFrame({'PC1':X_p[:,0], 'PC2':X_p[:,1],
        'Band':df['AIRI_band'].values, 'Score':df['AIRI_composite'].values,
        'Cluster':clusters.astype(str)})
    cmap = {'Nascent':'#cc0000','Developing':'#cc7700','Established':GREEN,'Advanced':'#0000aa'}
    fig = px.scatter(pdf, x='PC1', y='PC2', color='Band', size='Score',
                     hover_data=['Score'], color_discrete_map=cmap)
    fig.update_layout(
        title=dict(text=f"PCA Clustering  (PC1={pca.explained_variance_ratio_[0]:.0%}, PC2={pca.explained_variance_ratio_[1]:.0%} variance)",
                   font=dict(size=13, color=BLACK), x=0.5),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=400,
        xaxis=dict(tickfont=dict(color=BLACK), gridcolor='#eeeeee'),
        yaxis=dict(tickfont=dict(color=BLACK), gridcolor='#eeeeee'),
        legend=dict(font=dict(color=BLACK)))
    return fig

# ── SESSION STATE ──────────────────────────────────────────────────────────────
def init():
    for k, v in {'responses':{},'dim_scores':{},'composite':0,
                 'done':False,'page':'home'}.items():
        if k not in st.session_state:
            st.session_state[k] = v

def save(responses, dim_scores, comp):
    st.session_state.responses  = responses
    st.session_state.dim_scores = dim_scores
    st.session_state.composite  = comp
    st.session_state.done       = True
    band = classify(comp)
    ts   = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open('airi_responses.txt','a',encoding='utf-8') as f:
            f.write("=== AIRI Assessment | " + ts + " ===\n")
            f.write("Composite Score: " + str(round(comp,2)) + " | Band: " + band + "\n")
            for d, s in dim_scores.items():
                f.write("  " + d + ": " + str(round(s,2)) + "\n")
            f.write("Raw Responses: " + json.dumps(responses) + "\n")
            f.write("-"*50 + "\n\n")
    except Exception:
        pass

# ── PAGE: HOME ─────────────────────────────────────────────────────────────────
def pg_home():
    st.markdown("""
    <div style="background:#007700;padding:28px 24px;border-radius:6px;margin-bottom:20px;">
      <h1 style="color:#ffffff;margin:0;font-size:2rem;">🤖  AIRI — AI Readiness Index</h1>
      <p style="color:#ddffdd;margin:6px 0 0 0;font-size:1rem;">
        UK Debt Management Institutions &nbsp;|&nbsp;
        FCA Consumer Duty Aligned &nbsp;|&nbsp;
        Bournemouth University COMP7024 2025/26
      </p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    for col, icon, title, desc in zip(
        [col1, col2, col3],
        ["🔍","📊","💡"],
        ["Self-Assessment","Interactive Dashboard","Actionable Insights"],
        ["Complete a structured survey across 5 dimensions and 31 indicators to evaluate your organisation's AI readiness.",
         "Visualise your readiness profile with radar charts, gauges, and comparative analytics.",
         "Receive recommendations based on your readiness band and dimension-specific gap analysis."]):
        with col:
            st.markdown(f"""
            <div style="background:#f4f4f4;padding:18px;border-radius:6px;
                        border-top:3px solid #007700;height:100%;">
                <div style="font-size:1.8rem;margin-bottom:8px;">{icon}</div>
                <p style="font-weight:700;color:#111111;margin:0 0 6px 0;">{title}</p>
                <p style="color:#333333;font-size:0.88rem;margin:0;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, c2, _ = st.columns([1,2,1])
    with c2:
        if st.button("🚀  Start AIRI Assessment", use_container_width=True):
            st.session_state.page = 'survey'; st.rerun()

    st.markdown("---")
    st.markdown("## AIRI Framework Overview")

    cdims = get_conceptual_dims()
    bands = get_bands()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Conceptual Dimensions")
        for name, info in cdims.items():
            with st.expander(f"{info['code']}: {name}"):
                st.write(info['desc'])
                for ind in info['inds']:
                    st.write(f"• {ind}")
    with col2:
        st.markdown("### Readiness Bands")
        for name, info in bands.items():
            with st.expander(f"{name}  ({info['min']}–{info['max']})"):
                st.markdown(f"""
                <div style="background:{info['bg']};border-left:4px solid {info['border']};
                            padding:10px;border-radius:4px;margin-bottom:8px;">
                    <p style="color:{info['text']};font-weight:600;margin:0;">{info['desc']}</p>
                </div>""", unsafe_allow_html=True)
                for rec in info['recs'][:3]:
                    st.write(f"• {rec}")

# ── PAGE: SURVEY ───────────────────────────────────────────────────────────────
def pg_survey():
    st.markdown("# AIRI Expert Assessment Survey")
    st.markdown("""Rate your organisation's current capability for each indicator on a scale of **0 to 8**.
**0** = Not implemented &nbsp;|&nbsp; **4** = Partially implemented &nbsp;|&nbsp; **8** = Fully optimised""")
    st.markdown("---")

    questions = get_questions()
    responses = {}

    st.markdown("### Respondent Information")
    c1, c2, c3 = st.columns(3)
    with c1:
        role = st.selectbox("Your Role",
            ["Select...","C-Suite Executive","IT Director","Data Scientist",
             "Risk Manager","Compliance Officer","Operations Manager","Legal / Regulatory","Other"])
    with c2:
        experience = st.selectbox("Years in Role",
            ["Select...","< 1 year","1–3 years","3–5 years","5–10 years","> 10 years"])
    with c3:
        org_size = st.selectbox("Organisation Size",
            ["Select...","Small (< 50)","Medium (50–250)","Large (250–1000)","Enterprise (> 1000)"])

    st.markdown("<br>", unsafe_allow_html=True)

    for dim_name, qs in questions.items():
        st.markdown(f"""
        <div style="background:#f4f4f4;border-left:4px solid #007700;
                    padding:10px 14px;border-radius:4px;margin:14px 0 6px 0;">
            <p style="color:#111111;font-weight:700;font-size:1rem;margin:0;">{dim_name}</p>
            <p style="color:#444444;font-size:0.83rem;margin:3px 0 0 0;">
                Rate each indicator from 0 (no capability) to 8 (fully optimised)
            </p>
        </div>""", unsafe_allow_html=True)

        for q in qs:
            c1, c2 = st.columns([3,1])
            with c1:
                st.markdown(f"""
                <div style="padding:9px 12px;background:#ffffff;border:1px solid #dddddd;
                            border-radius:4px;margin:3px 0;">
                    <p style="color:#007700;font-weight:700;font-size:0.85rem;margin:0 0 2px 0;">
                        {q['id']} — {q['sub']}
                    </p>
                    <p style="color:#111111;font-size:0.87rem;margin:0;">{q['text']}</p>
                </div>""", unsafe_allow_html=True)
            with c2:
                responses[q['id']] = st.slider(
                    "Score", 0, 8, 0,
                    key=f"s_{q['id']}",
                    label_visibility="collapsed")

        st.markdown("<br>", unsafe_allow_html=True)

    _, c2, _ = st.columns([1,2,1])
    with c2:
        if st.button("📊  Calculate My AIRI Score", use_container_width=True, type="primary"):
            if "Select..." in [role, experience, org_size]:
                st.error("Please complete all respondent information fields.")
            else:
                ds   = {d: dim_score(responses, qs) for d, qs in questions.items()}
                comp = composite(ds)
                save(responses, ds, comp)
                st.session_state.page = 'results'
                st.rerun()

    if st.button("← Back to Home"):
        st.session_state.page = 'home'; st.rerun()

# ── PAGE: RESULTS ──────────────────────────────────────────────────────────────
def pg_results():
    if not st.session_state.get('done', False):
        st.warning("No assessment data found. Please complete the survey first.")
        if st.button("Go to Survey"):
            st.session_state.page = 'survey'; st.rerun()
        return

    ds   = st.session_state.dim_scores
    comp = st.session_state.composite
    band = classify(comp)
    bi   = get_bands()[band]
    best = max(ds, key=ds.get)
    worst= min(ds, key=ds.get)

    st.markdown(f"""
    <div style="background:#007700;padding:22px;border-radius:6px;margin-bottom:18px;">
        <h1 style="color:#ffffff;margin:0;">Your AIRI Assessment Results</h1>
        <p style="color:#ddffdd;margin:5px 0 0 0;">
            Completed on {datetime.now().strftime('%d %B %Y')}
        </p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    def kpi(col, bg, border, tc, label, value, sub):
        col.markdown(f"""
        <div style="background:{bg};border:2px solid {border};padding:14px;
                    border-radius:6px;text-align:center;">
            <p style="color:{tc};font-size:0.75rem;text-transform:uppercase;
               letter-spacing:1px;margin:0;">{label}</p>
            <p style="color:{tc};font-size:2.2rem;font-weight:700;margin:4px 0;">{value}</p>
            <p style="color:{tc};font-size:0.78rem;margin:0;">{sub}</p>
        </div>""", unsafe_allow_html=True)

    kpi(c1, "#007700", "#007700", WHITE,  "Composite Score", f"{comp:.1f}", "out of 100")
    kpi(c2, bi['bg'],  bi['border'], bi['text'], "Readiness Band",  band, f"{bi['min']}–{bi['max']}")
    kpi(c3, "#eeffee", "#007700",  "#005500", "Strongest", best, f"{ds[best]:.1f}/100")
    kpi(c4, "#ffeeee", "#cc0000",  "#cc0000", "Weakest",  worst, f"{ds[worst]:.1f}/100")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.plotly_chart(ch_gauge(comp),   use_container_width=True)
    with c2: st.plotly_chart(ch_radar(ds),      use_container_width=True)

    st.markdown("## Dimension Score Breakdown")
    st.plotly_chart(ch_bar(ds), use_container_width=True)

    st.markdown("## Dimension Impact Analysis")
    st.plotly_chart(ch_impact(ds), use_container_width=True)

    st.markdown(f"""
    <div style="background:{bi['bg']};border-left:5px solid {bi['border']};
                padding:18px;border-radius:5px;margin:14px 0;">
        <h3 style="color:{bi['text']};margin:0 0 6px 0;">
            Recommendations for {band} Organisations
        </h3>
        <p style="color:#222222;margin:0;">{bi['desc']}</p>
    </div>""", unsafe_allow_html=True)

    st.write("**Priority Actions:**")
    for i, rec in enumerate(bi['recs'], 1):
        st.write(f"{i}. {rec}")

    st.markdown("## Gap Analysis")
    gap_df = pd.DataFrame([{
        'Dimension':     d,
        'Current Score': f"{s:.1f}",
        'Gap':           f"{100-s:.1f}",
        'Priority':      'High' if (100-s)>60 else 'Medium' if (100-s)>40 else 'Low'
    } for d, s in ds.items()])
    st.dataframe(gap_df, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("← Retake Assessment", use_container_width=True):
            st.session_state.page = 'survey'; st.rerun()
    with c2:
        if st.button("📈  Analytics Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()
    with c3:
        csv_data = pd.DataFrame(
            [{'Dimension':k, 'Score':v} for k,v in ds.items()] +
            [{'Dimension':'AIRI Composite', 'Score':comp}]
        ).to_csv(index=False)
        st.download_button("📥  Export CSV", csv_data,
                           "airi_results.csv", "text/csv",
                           use_container_width=True)

# ── PAGE: DASHBOARD ────────────────────────────────────────────────────────────
def pg_dashboard():
    st.markdown(f"""
    <div style="background:#007700;padding:22px;border-radius:6px;margin-bottom:18px;">
        <h1 style="color:#ffffff;margin:0;">AIRI Aggregate Analytics Dashboard</h1>
        <p style="color:#ddffdd;margin:5px 0 0 0;">Comparative analysis across all assessed organisations</p>
    </div>""", unsafe_allow_html=True)

    all_recs = []
    try:
        if os.path.exists('airi_responses.txt'):
            with open('airi_responses.txt','r',encoding='utf-8') as f:
                recs = f.read().split('=== AIRI Assessment | ')
                for rec in recs[1:]:
                    lines = rec.strip().split('\n')
                    comp  = float(lines[1].split('|')[0].split(':')[1].strip())
                    band  = lines[1].split('|')[1].split(':')[1].strip()
                    ds    = {}
                    for ln in lines[2:]:
                        if ln.startswith('  ') and ':' in ln and 'Raw' not in ln:
                            parts = ln.strip().split(':')
                            try: ds[parts[0].strip()] = float(parts[1].strip())
                            except: pass
                    all_recs.append({**ds, 'AIRI_composite':comp, 'AIRI_band':band})
    except Exception:
        pass

    if not all_recs:
        st.info("No assessment data yet. Complete a survey to populate this dashboard.")
        if st.button("← Go to Survey"):
            st.session_state.page = 'survey'; st.rerun()
        return

    df = pd.DataFrame(all_recs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assessments",  len(df))
    c2.metric("Mean AIRI Score",    f"{df['AIRI_composite'].mean():.1f}")
    c3.metric("Median AIRI Score",  f"{df['AIRI_composite'].median():.1f}")
    c4.metric("Most Common Band",   df['AIRI_band'].mode()[0] if not df.empty else "N/A")

    dim_cols = [c for c in df.columns if c not in ['AIRI_composite','AIRI_band']]
    if dim_cols:
        avg = df[dim_cols].mean().to_dict()
        st.markdown("## Average Dimension Profile")
        st.plotly_chart(ch_radar(avg, "Average Dimension Profile (All Assessments)"),
                        use_container_width=True)
        st.markdown("## PCA Clustering")
        st.plotly_chart(ch_pca(df), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.page = 'results'; st.rerun()
    with c2:
        if st.button("🏠  Home", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:14px 0 8px 0;
                    border-bottom:1px solid #cccccc;margin-bottom:12px;">
            <p style="font-size:1.5rem;font-weight:700;margin:0;">🤖 AIRI</p>
            <p style="font-size:0.82rem;color:#555555;margin:2px 0 0 0;">AI Readiness Index</p>
        </div>""", unsafe_allow_html=True)

        nav = [("🏠  Home",'home'),("📝  Take Assessment",'survey'),
               ("📊  My Results",'results'),("📈  Analytics Dashboard",'dashboard')]
        for label, key in nav:
            if st.button(label, use_container_width=True, key=f"nav_{key}"):
                if key == 'results' and not st.session_state.get('done'):
                    st.warning("Complete an assessment first.")
                else:
                    st.session_state.page = key; st.rerun()

        st.markdown("---")
        st.markdown("""
        <div style="background:#f0f0f0;padding:12px;border-radius:4px;
                    border-left:3px solid #007700;">
            <p style="font-weight:700;font-size:0.88rem;margin:0 0 5px 0;">About AIRI</p>
            <p style="font-size:0.8rem;color:#333333;margin:0;line-height:1.5;">
                Quantitative diagnostic tool for UK debt management institutions.
                Aligned with FCA Consumer Duty.
                Validated by expert survey (n=121).
            </p>
        </div>
        <br>
        <p style="font-size:0.75rem;color:#555555;text-align:center;margin:0;">
            Bournemouth University<br>
            MSc IT | COMP7024 | 2025/26<br>
            Sedara Aanuoluwapo Endurance
        </p>""", unsafe_allow_html=True)

# ── MAIN ───────────────────────────────────────────────────────────────────────
init()
sidebar()

page = st.session_state.page
if   page == 'home':      pg_home()
elif page == 'survey':    pg_survey()
elif page == 'results':   pg_results()
elif page == 'dashboard': pg_dashboard()
else:                     pg_home()
