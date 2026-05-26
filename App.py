"""
=============================================================================
AIRI (AI Readiness Index) Interactive Assessment Artifact
For: UK Debt Management Institutions
Built with Streamlit
Bournemouth University | MSc Information Technology | COMP7024 | 2025/26
Sedara Aanuoluwapo Endurance
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


def configure_page():
    st.set_page_config(
        page_title="AIRI - AI Readiness Index for UK Debt Management",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stApp { background-color: #f8f9fa; }
        h1, h2, h3, h4, h5, h6 { font-family: 'Segoe UI', sans-serif; }
        h1 { color: #1e3a5f !important; font-weight: 700; }
        h2 { color: #2c5282 !important; font-weight: 600;
             border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }
        h3 { color: #2d3748 !important; font-weight: 600; }
        p, li, span, div { color: inherit; }
        .stMarkdown p { color: #2d3748; }
        .stMarkdown li { color: #2d3748; }
        label { color: #2d3748 !important; }
        .stSlider label { color: #2d3748 !important; }
        .stSelectbox label { color: #2d3748 !important; }
        [data-testid="stSidebar"] { background-color: #1e3a5f; }
        [data-testid="stSidebar"] p { color: #e2e8f0 !important; }
        [data-testid="stSidebar"] label { color: #e2e8f0 !important; }
        [data-testid="stSidebar"] .stButton>button { background-color: rgba(255,255,255,0.15);
            color: white !important; border: 1px solid rgba(255,255,255,0.3); }
        [data-testid="stSidebar"] .stButton>button:hover { background-color: rgba(255,255,255,0.25); }
        .stButton>button { background-color: #2c5282; color: white !important;
            border-radius: 8px; padding: 10px 24px; font-weight: 600; border: none; }
        .stButton>button:hover { background-color: #1e3a5f;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white !important; padding: 20px; border-radius: 12px; text-align: center; }
        .metric-card * { color: white !important; }
        .metric-value { font-size: 2.5em; font-weight: bold; margin: 10px 0; color: white !important; }
        .metric-label { font-size: 0.9em; opacity: 0.9; text-transform: uppercase;
            letter-spacing: 1px; color: white !important; }
        .stExpander { background: white; border-radius: 8px; }
        .stExpander p { color: #4a5568 !important; }
        .stDataFrame { background: white; }
        .stMetric label { color: #4a5568 !important; }
        .stMetric [data-testid="stMetricValue"] { color: #1e3a5f !important; }
        .stWarning p { color: #744210 !important; }
        .stError p { color: #742a2a !important; }
        .stSuccess p { color: #276749 !important; }
    </style>
    """, unsafe_allow_html=True)


def get_airi_dimensions():
    conceptual_dimensions = {
        "Data Infrastructure": {
            "code": "D1",
            "description": "Quality, integration, governance, and security of data assets",
            "indicators": ["Data quality standards and monitoring","Data integration across systems",
                           "Data governance framework","Data security and privacy controls","Real-time data availability"]
        },
        "Technological Maturity": {
            "code": "D2",
            "description": "Technical infrastructure and MLOps capabilities",
            "indicators": ["Cloud infrastructure readiness","MLOps and model deployment pipelines",
                           "API and integration architecture","Computational resources for AI","Monitoring and observability tools"]
        },
        "Regulatory Compliance": {
            "code": "D3",
            "description": "FCA compliance, data protection, cybersecurity, vulnerability protocols",
            "indicators": ["FCA Consumer Duty alignment","GDPR and data protection compliance",
                           "Cybersecurity framework","Vulnerable customer protocols","Audit trail and documentation"]
        },
        "Organisational Capability": {
            "code": "D4",
            "description": "Leadership, skills, culture, and resources for AI adoption",
            "indicators": ["Executive AI leadership and sponsorship","AI literacy and training programs",
                           "Cross-functional AI teams","Change management capability","Budget and resource allocation"]
        },
        "Ethical Governance": {
            "code": "D5",
            "description": "Bias mitigation, fairness, transparency, and consumer protection",
            "indicators": ["Algorithmic bias detection and mitigation","Explainability and interpretability frameworks",
                           "Fairness assessment procedures","Consumer protection safeguards","Ethical review board or committee"]
        }
    }
    analytical_dimensions = {
        "Strategy_Governance": {"description":"AI strategy, decision rights, oversight, and FCA alignment","weight":0.25,"color":"#2c5282"},
        "Data_Technology":      {"description":"Data infrastructure, MLOps, integration, and security",  "weight":0.25,"color":"#38a169"},
        "People_Skills":        {"description":"Leadership, skills, culture, and resources",              "weight":0.25,"color":"#d69e2e"},
        "Risk_Ethics":          {"description":"Bias mitigation, explainability, accountability, Consumer Duty","weight":0.25,"color":"#e53e3e"},
    }
    return conceptual_dimensions, analytical_dimensions


def get_readiness_bands():
    return {
        "Nascent":     {"min":0,  "max":25,  "color":"#fc8181","dark_color":"#c53030",
                        "description":"Fundamental gaps in AI readiness. Early-stage exploration with limited infrastructure, governance, or skills.",
                        "recommendations":["Establish basic data governance policies","Develop initial AI strategy and roadmap",
                                           "Conduct AI literacy training for leadership","Review FCA Consumer Duty requirements","Create ethical AI principles document"]},
        "Developing":  {"min":26, "max":50,  "color":"#fbd38d","dark_color":"#c05621",
                        "description":"Building core capabilities. Partial implementation with identified gaps in technology, governance, or workforce.",
                        "recommendations":["Implement data quality monitoring tools","Establish MLOps pipelines for model deployment",
                                           "Expand AI training to operational staff","Develop vulnerability-sensitive customer protocols","Create model explainability frameworks"]},
        "Established": {"min":51, "max":75,  "color":"#9ae6b4","dark_color":"#276749",
                        "description":"Mature AI practices. Operational AI with robust governance, compliance, and continuous monitoring.",
                        "recommendations":["Optimise AI model performance and drift monitoring","Enhance cross-functional AI governance committees",
                                           "Implement advanced bias detection algorithms","Develop peer benchmarking capabilities","Create automated compliance reporting"]},
        "Advanced":    {"min":76, "max":100, "color":"#90cdf4","dark_color":"#2c5282",
                        "description":"Leading-edge AI readiness. Continuous innovation, proactive governance, and industry-leading practices.",
                        "recommendations":["Pioneer new AI governance standards","Develop AI-driven regulatory horizon scanning",
                                           "Create industry collaboration frameworks","Implement real-time ethical AI monitoring","Establish AI research and innovation labs"]},
    }


def generate_survey_questions():
    return {
        "Strategy_Governance": [
            {"id":"Q27","text":"Our organisation has a documented AI strategy aligned with business objectives and FCA expectations.","sub_dimension":"AI Strategy"},
            {"id":"Q28","text":"Clear decision rights and accountability structures exist for AI system deployment and oversight.","sub_dimension":"Decision Rights"},
            {"id":"Q29","text":"We have established AI governance committees with cross-functional representation (IT, Risk, Compliance, Legal).","sub_dimension":"Governance Structure"},
            {"id":"Q30","text":"Our AI governance framework explicitly addresses the FCA Consumer Duty requirements for fair outcomes.","sub_dimension":"FCA Alignment"},
            {"id":"Q31","text":"We maintain comprehensive audit trails for all AI-driven decisions affecting customers.","sub_dimension":"Audit & Accountability"},
            {"id":"Q32","text":"Regular board-level reviews of AI risks, performance, and strategic alignment are conducted.","sub_dimension":"Board Oversight"},
            {"id":"Q33","text":"Our organisation has defined AI risk appetite statements integrated into enterprise risk management.","sub_dimension":"Risk Appetite"},
            {"id":"Q34","text":"We have established clear escalation procedures for AI incidents and customer complaints.","sub_dimension":"Incident Management"},
        ],
        "Data_Technology": [
            {"id":"Q11","text":"Our data infrastructure supports real-time or near-real-time data processing for AI applications.","sub_dimension":"Data Infrastructure"},
            {"id":"Q12","text":"Data quality is systematically monitored with defined metrics and remediation procedures.","sub_dimension":"Data Quality"},
            {"id":"Q13","text":"We have implemented data lineage tracking to understand data provenance for AI model inputs.","sub_dimension":"Data Lineage"},
            {"id":"Q14","text":"Our organisation has cloud-based or scalable on-premise infrastructure for AI model training and deployment.","sub_dimension":"Cloud Infrastructure"},
            {"id":"Q15","text":"MLOps practices (version control, CI/CD, model registry) are implemented for AI lifecycle management.","sub_dimension":"MLOps"},
            {"id":"Q16","text":"APIs and integration layers enable seamless data flow between operational systems and AI platforms.","sub_dimension":"Integration"},
            {"id":"Q17","text":"We have adequate computational resources (GPU/TPU) for training and inference of AI models.","sub_dimension":"Compute Resources"},
            {"id":"Q18","text":"Data security controls (encryption, access controls, anonymization) meet financial services standards.","sub_dimension":"Data Security"},
            {"id":"Q19","text":"Monitoring and observability tools track AI model performance, drift, and operational health.","sub_dimension":"Monitoring"},
            {"id":"Q20","text":"Our data architecture supports integration of structured and unstructured data for AI applications.","sub_dimension":"Data Architecture"},
        ],
        "People_Skills": [
            {"id":"Q21","text":"Executive leadership demonstrates visible commitment and sponsorship for AI initiatives.","sub_dimension":"Leadership"},
            {"id":"Q22","text":"We have conducted AI literacy assessments and identified skill gaps across the organisation.","sub_dimension":"AI Literacy"},
            {"id":"Q23","text":"Role-based AI training programs are available for technical and non-technical staff.","sub_dimension":"Training Programs"},
            {"id":"Q24","text":"Cross-functional AI teams (data scientists, engineers, domain experts) are established and resourced.","sub_dimension":"Team Structure"},
            {"id":"Q25","text":"Change management processes support AI adoption and address workforce transition concerns.","sub_dimension":"Change Management"},
            {"id":"Q26","text":"We have access to external AI expertise (consultants, vendors, academic partnerships) when needed.","sub_dimension":"External Expertise"},
        ],
        "Risk_Ethics": [
            {"id":"Q35","text":"We have implemented procedures to detect and mitigate algorithmic bias in AI models.","sub_dimension":"Bias Mitigation"},
            {"id":"Q36","text":"AI model decisions can be explained to regulators, customers, and internal stakeholders.","sub_dimension":"Explainability"},
            {"id":"Q37","text":"Fairness assessments are conducted regularly to ensure equitable outcomes across customer segments.","sub_dimension":"Fairness"},
            {"id":"Q38","text":"Our AI systems incorporate vulnerability-sensitive design for financially distressed customers.","sub_dimension":"Vulnerability Sensitivity"},
            {"id":"Q39","text":"Human-in-the-loop protocols ensure meaningful oversight of high-stakes AI decisions.","sub_dimension":"Human Oversight"},
            {"id":"Q40","text":"Ethical review processes evaluate AI use cases before deployment.","sub_dimension":"Ethical Review"},
            {"id":"Q41","text":"We have documented procedures for handling AI-related customer complaints and remediation.","sub_dimension":"Consumer Protection"},
        ],
    }


def normalize_score(raw_score, min_val=0, max_val=8):
    if max_val == min_val:
        return 50.0
    return max(0, min(100, ((raw_score - min_val) / (max_val - min_val)) * 100))


def compute_dimension_score(responses, dimension_questions):
    scores = [normalize_score(responses[q["id"]]) for q in dimension_questions if q["id"] in responses]
    return np.mean(scores) if scores else 0.0


def compute_composite_airi(dimension_scores, weights=None):
    if weights is None:
        weights = {dim: 0.25 for dim in dimension_scores}
    return sum(dimension_scores[dim] * weights.get(dim, 0.25) for dim in dimension_scores)


def classify_readiness_band(score):
    bands = get_readiness_bands()
    for band_name, band_info in bands.items():
        if band_name == "Advanced":
            if band_info["min"] <= score <= band_info["max"]:
                return band_name
        else:
            if band_info["min"] <= score < band_info["max"]:
                return band_name
    return "Unknown"


def create_radar_chart(dimension_scores, title="AIRI Dimension Profile"):
    categories = list(dimension_scores.keys())
    values = list(dimension_scores.values())
    cats_closed = categories + [categories[0]]
    vals_closed = values + [values[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals_closed, theta=cats_closed, fill='toself',
        fillcolor='rgba(44,82,130,0.3)', line=dict(color='#2c5282',width=3), name='Current Score'))
    for val, label in [(25,'Nascent'),(50,'Developing'),(75,'Established')]:
        fig.add_trace(go.Scatterpolar(r=[val]*(len(categories)+1), theta=cats_closed, mode='lines',
            line=dict(dash='dash',width=1,color='gray'), name=label+' Threshold', opacity=0.5))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,100],tickvals=[0,25,50,75,100])),
        showlegend=True, legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5),
        title=dict(text=title,font=dict(size=20,color='#1e3a5f'),x=0.5),
        paper_bgcolor='white', height=500)
    return fig


def create_dimension_bar_chart(dimension_scores):
    dimensions = list(dimension_scores.keys())
    scores = list(dimension_scores.values())
    colors = ['#fc8181' if s<25 else '#fbd38d' if s<50 else '#9ae6b4' if s<75 else '#90cdf4' for s in scores]
    fig = go.Figure(data=[go.Bar(x=scores, y=dimensions, orientation='h',
        marker=dict(color=colors, line=dict(color='white',width=2)),
        text=[f'{s:.1f}' for s in scores], textposition='auto',
        textfont=dict(size=14,color='white',family='Arial Black'))])
    for val, label, color in [(25,'Nascent','#c53030'),(50,'Developing','#c05621'),(75,'Established','#276749')]:
        fig.add_vline(x=val, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="top")
    fig.update_layout(xaxis=dict(range=[0,100],title=dict(text="Score (0-100)")),
        title=dict(text="Dimension Score Breakdown",font=dict(size=18,color='#1e3a5f'),x=0.5),
        plot_bgcolor='white', paper_bgcolor='white', height=400, margin=dict(l=150))
    return fig


def create_gauge_chart(score, title="AIRI Composite Score"):
    band = classify_readiness_band(score)
    band_info = get_readiness_bands()[band]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=score,
        number={'suffix':"/100",'font':{'size':48,'color':band_info['dark_color']}},
        title={'text':title,'font':{'size':20,'color':'#1e3a5f'}},
        delta={'reference':50,'position':"bottom"},
        gauge={'axis':{'range':[0,100]},'bar':{'color':band_info['dark_color'],'thickness':0.75},
               'bgcolor':'white','borderwidth':2,'bordercolor':'#e2e8f0',
               'steps':[{'range':[0,25],'color':'#fed7d7'},{'range':[25,50],'color':'#feebc8'},
                        {'range':[50,75],'color':'#c6f6d5'},{'range':[75,100],'color':'#bee3f8'}],
               'threshold':{'line':{'color':'black','width':4},'thickness':0.8,'value':score}}))
    fig.update_layout(height=400, paper_bgcolor='white', margin=dict(t=80,b=20))
    return fig


def create_shap_style_importance(dimension_scores):
    dimensions = list(dimension_scores.keys())
    scores = list(dimension_scores.values())
    mean_score = np.mean(scores)
    shap_values = [s - mean_score for s in scores]
    colors = ['#e53e3e' if v < 0 else '#38a169' for v in shap_values]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=dimensions, x=shap_values, orientation='h',
        marker=dict(color=colors), text=[f'{v:+.1f}' for v in shap_values], textposition='outside'))
    fig.add_vline(x=0, line_color='black', line_width=1)
    fig.update_layout(
        title=dict(text="Dimension Impact on Readiness (Deviation from Mean)",font=dict(size=18,color='#1e3a5f'),x=0.5),
        xaxis=dict(title=dict(text="Impact on Composite Score"),zeroline=True,zerolinecolor='black'),
        yaxis=dict(tickfont=dict(size=13)), plot_bgcolor='white', paper_bgcolor='white',
        height=350, margin=dict(l=150), showlegend=False)
    return fig


def create_pca_scatter(all_responses_df):
    if len(all_responses_df) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for PCA (need 3+ responses)", showarrow=False, font=dict(size=16,color='#718096'))
        fig.update_layout(height=400)
        return fig
    dim_cols = ['Strategy_Governance','Data_Technology','People_Skills','Risk_Ethics']
    X = all_responses_df[dim_cols].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    n_clusters = min(4, len(all_responses_df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    plot_df = pd.DataFrame({'PC1':X_pca[:,0],'PC2':X_pca[:,1],'Cluster':clusters,
        'Band':all_responses_df['AIRI_band'].values,'Score':all_responses_df['AIRI_composite'].values})
    fig = px.scatter(plot_df, x='PC1', y='PC2', color='Band', symbol='Cluster', size='Score',
        hover_data=['Score'],
        color_discrete_map={'Nascent':'#fc8181','Developing':'#fbd38d','Established':'#9ae6b4','Advanced':'#90cdf4'})
    fig.update_layout(
        title=dict(text="PCA: Readiness Structure & Clustering",font=dict(size=18,color='#1e3a5f'),x=0.5),
        xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)"),
        yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)"),
        plot_bgcolor='white', paper_bgcolor='white', height=500)
    return fig


def init_session_state():
    defaults = {'responses':{},'dimension_scores':{},'composite_score':0,
                'assessment_complete':False,'all_responses':[],'page':'home'}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def save_response_to_session(responses, dimension_scores, composite_score):
    band = classify_readiness_band(composite_score)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    st.session_state.responses = responses
    st.session_state.dimension_scores = dimension_scores
    st.session_state.composite_score = composite_score
    st.session_state.assessment_complete = True
    try:
        with open('airi_responses.txt', 'a', encoding='utf-8') as f:
            f.write("=== AIRI Assessment | " + timestamp + " ===\n")
            f.write("Composite Score: " + str(round(composite_score,2)) + " | Band: " + band + "\n")
            for dim, score in dimension_scores.items():
                f.write("  " + dim + ": " + str(round(score,2)) + "\n")
            f.write("Raw Responses: " + json.dumps(responses) + "\n")
            f.write("-" * 50 + "\n\n")
    except Exception:
        pass


def clear_assessment():
    st.session_state.responses = {}
    st.session_state.dimension_scores = {}
    st.session_state.composite_score = 0
    st.session_state.assessment_complete = False


def render_home():
    st.markdown("""
    <div style="text-align:center;padding:40px 20px;background:linear-gradient(135deg,#1e3a5f 0%,#2c5282 100%);border-radius:15px;margin-bottom:30px;">
        <h1 style="color:white;font-size:3em;margin-bottom:10px;">🤖 AIRI</h1>
        <h2 style="color:#bee3f8;font-size:1.5em;font-weight:400;border:none;">AI Readiness Index for UK Debt Management</h2>
        <p style="color:#e2e8f0;font-size:1.1em;max-width:700px;margin:20px auto;">
            A quantitative diagnostic tool to assess institutional preparedness for ethical AI agent deployment
            in UK debt management institutions, aligned with FCA Consumer Duty requirements.
        </p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    cards = [
        ("🔍","Self-Assessment","Complete a structured survey across 5 dimensions and 31 indicators to evaluate your organisation's AI readiness."),
        ("📊","Interactive Dashboard","Visualize your readiness profile with radar charts, gauges, and comparative analytics against industry benchmarks."),
        ("💡","Actionable Insights","Receive personalized recommendations based on your readiness band and dimension-specific gap analysis."),
    ]
    for col, (icon, title, desc) in zip([col1,col2,col3], cards):
        with col:
            st.markdown(f"""
            <div style="background:white;padding:25px;border-radius:12px;box-shadow:0 4px 15px rgba(0,0,0,0.08);height:100%;">
                <div style="font-size:2.5em;text-align:center;margin-bottom:15px;">{icon}</div>
                <h3 style="text-align:center;color:#2c5282;">{title}</h3>
                <p style="color:#4a5568;text-align:center;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("🚀 Start AIRI Assessment", use_container_width=True):
            st.session_state.page = 'survey'
            st.rerun()

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown('<h2 style="text-align:center;color:#1e3a5f;">AIRI Framework Overview</h2>', unsafe_allow_html=True)
    conceptual_dims, _ = get_airi_dimensions()
    bands = get_readiness_bands()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Conceptual Dimensions")
        for name, info in conceptual_dims.items():
            with st.expander(f"{info['code']}: {name}"):
                st.write(info['description'])
                for ind in info['indicators']:
                    st.write(f"• {ind}")
    with col2:
        st.subheader("Readiness Bands")
        for name, info in bands.items():
            with st.expander(f"{name} ({info['min']}–{info['max']})"):
                st.markdown(f'<div style="background:{info["color"]};padding:10px;border-radius:8px;color:{info["dark_color"]};"><strong>{info["description"]}</strong></div>', unsafe_allow_html=True)
                for rec in info['recommendations'][:3]:
                    st.write(f"• {rec}")


def render_survey():
    st.markdown('<h1 style="color:#1e3a5f;">AIRI Expert Assessment Survey</h1>', unsafe_allow_html=True)
    st.markdown("Rate your organisation's current capability for each indicator on a scale of **0–8** (0 = Not implemented, 8 = Fully optimised / Industry leading).")
    st.markdown("---")

    questions = generate_survey_questions()
    responses = {}

    st.subheader("👤 Respondent Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        role = st.selectbox("Your Role", ["Select...","C-Suite Executive","IT/Director","Data Scientist",
                                          "Risk Manager","Compliance Officer","Operations Manager","Legal/Regulatory","Other"])
    with col2:
        experience = st.selectbox("Years in Role", ["Select...","< 1 year","1-3 years","3-5 years","5-10 years","> 10 years"])
    with col3:
        org_size = st.selectbox("Organisation Size", ["Select...","Small (< 50 staff)","Medium (50-250)","Large (250-1000)","Enterprise (> 1000)"])

    st.markdown("<br>", unsafe_allow_html=True)

    for dim_name, dim_questions in questions.items():
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#ebf8ff 0%,#bee3f8 100%);padding:15px;border-radius:10px;margin:20px 0;">
            <h3 style="color:#2c5282;margin:0;">{dim_name.replace('_',' ')}</h3>
            <p style="color:#4a5568;margin:5px 0 0 0;font-size:0.9em;">Rate each indicator from 0 (no capability) to 8 (fully optimised)</p>
        </div>""", unsafe_allow_html=True)

        for q in dim_questions:
            col1, col2 = st.columns([3,1])
            with col1:
                st.markdown(f"""
                <div style="padding:10px;background:white;border-radius:8px;margin:5px 0;">
                    <strong>{q['id']}</strong>: {q['text']}<br>
                    <span style="color:#718096;font-size:0.85em;">{q['sub_dimension']}</span>
                </div>""", unsafe_allow_html=True)
            with col2:
                responses[q['id']] = st.slider("Score", min_value=0, max_value=8, value=0, key=f"survey_{q['id']}", help="0=Not implemented, 8=Fully optimised")
        st.markdown("<hr style='margin:30px 0;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        if st.button("📊 Calculate AIRI Score", use_container_width=True, type="primary"):
            if "Select..." in [role, experience, org_size]:
                st.error("Please complete all respondent information fields.")
            else:
                dimension_scores = {dim: compute_dimension_score(responses, qs) for dim, qs in questions.items()}
                composite = compute_composite_airi(dimension_scores)
                save_response_to_session(responses, dimension_scores, composite)
                st.session_state.page = 'results'
                st.rerun()

    if st.button("← Back to Home"):
        st.session_state.page = 'home'
        st.rerun()


def render_results():
    if not st.session_state.get('assessment_complete', False):
        st.warning("No assessment data found. Please complete the survey first.")
        if st.button("Go to Survey"):
            st.session_state.page = 'survey'
            st.rerun()
        return

    dimension_scores = st.session_state.dimension_scores
    composite_score = st.session_state.composite_score
    band = classify_readiness_band(composite_score)
    band_info = get_readiness_bands()[band]

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1e3a5f 0%,#2c5282 100%);padding:30px;border-radius:15px;margin-bottom:30px;">
        <h1 style="color:white;margin:0;">Your AIRI Assessment Results</h1>
        <p style="color:#bee3f8;font-size:1.1em;margin:10px 0 0 0;">Assessment completed on {datetime.now().strftime('%B %d, %Y')}</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    strongest = max(dimension_scores, key=dimension_scores.get)
    weakest   = min(dimension_scores, key=dimension_scores.get)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Composite Score</div><div class="metric-value">{composite_score:.1f}</div><div style="font-size:0.9em;opacity:0.8;">out of 100</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,{band_info["dark_color"]} 0%,{band_info["color"]} 100%);"><div class="metric-label">Readiness Band</div><div class="metric-value">{band}</div><div style="font-size:0.9em;opacity:0.8;">{band_info["min"]}–{band_info["max"]} range</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#38a169 0%,#48bb78 100%);"><div class="metric-label">Strongest Dimension</div><div class="metric-value" style="font-size:1.5em;">{strongest.replace("_"," ")}</div><div style="font-size:0.9em;opacity:0.8;">{dimension_scores[strongest]:.1f}/100</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card" style="background:linear-gradient(135deg,#e53e3e 0%,#fc8181 100%);"><div class="metric-label">Weakest Dimension</div><div class="metric-value" style="font-size:1.5em;">{weakest.replace("_"," ")}</div><div style="font-size:0.9em;opacity:0.8;">{dimension_scores[weakest]:.1f}/100</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_gauge_chart(composite_score), use_container_width=True)
    with col2:
        st.plotly_chart(create_radar_chart(dimension_scores), use_container_width=True)

    st.subheader("Dimension Score Breakdown")
    st.plotly_chart(create_dimension_bar_chart(dimension_scores), use_container_width=True)

    st.subheader("Dimension Impact Analysis (Deviation from Mean)")
    st.plotly_chart(create_shap_style_importance(dimension_scores), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,{band_info['color']} 0%,white 100%);padding:25px;border-radius:12px;border-left:5px solid {band_info['dark_color']};">
        <h3 style="color:{band_info['dark_color']};margin-top:0;">Recommendations for {band} Organisations</h3>
        <p style="color:#4a5568;font-size:1.05em;">{band_info['description']}</p>
    </div>""", unsafe_allow_html=True)
    st.write("**Priority Actions:**")
    for i, rec in enumerate(band_info['recommendations'], 1):
        st.write(f"{i}. {rec}")

    st.subheader("Gap Analysis")
    gap_data = [{'Dimension':d.replace('_',' '),'Current Score':s,'Gap':round(100-s,1),
                 'Priority':'High' if (100-s)>60 else 'Medium' if (100-s)>40 else 'Low'}
                for d, s in dimension_scores.items()]
    st.dataframe(pd.DataFrame(gap_data), use_container_width=True, hide_index=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("← Retake Assessment", use_container_width=True):
            st.session_state.page = 'survey'; st.rerun()
    with col2:
        if st.button("📊 View Aggregate Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()
    with col3:
        results_df = pd.DataFrame([{'Dimension':k.replace('_',' '),'Score':v} for k,v in dimension_scores.items()] + [{'Dimension':'AIRI Composite','Score':composite_score}])
        st.download_button("📥 Export Results (CSV)", results_df.to_csv(index=False), "airi_results.csv", "text/csv", use_container_width=True)


def render_dashboard():
    st.markdown("""
    <div style="background:linear-gradient(135deg,#2c5282 0%,#1e3a5f 100%);padding:30px;border-radius:15px;margin-bottom:30px;">
        <h1 style="color:white;margin:0;">AIRI Aggregate Analytics Dashboard</h1>
        <p style="color:#bee3f8;font-size:1.1em;margin:10px 0 0 0;">Comparative analysis across all assessed organisations</p>
    </div>""", unsafe_allow_html=True)

    all_responses = []
    try:
        if os.path.exists('airi_responses.txt'):
            with open('airi_responses.txt','r',encoding='utf-8') as f:
                records = f.read().split('=== AIRI Assessment | ')
                for rec in records[1:]:
                    lines = rec.strip().split('\n')
                    composite = float(lines[1].split('|')[0].split(':')[1].strip())
                    band = lines[1].split('|')[1].split(':')[1].strip()
                    dim_scores = {}
                    for line in lines[2:]:
                        if line.startswith('  ') and ':' in line and 'Raw' not in line:
                            parts = line.strip().split(':')
                            try: dim_scores[parts[0].strip()] = float(parts[1].strip())
                            except: pass
                    all_responses.append({'respondent_id':"ID_"+str(len(all_responses)+1).zfill(3),
                        **dim_scores,'AIRI_composite':composite,'AIRI_band':band})
    except Exception:
        pass

    if not all_responses:
        st.warning("No assessment data available yet. Complete a survey to see analytics.")
        if st.button("← Go to Survey"):
            st.session_state.page = 'survey'; st.rerun()
        return

    df = pd.DataFrame(all_responses)
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Assessments", len(df))
    col2.metric("Mean AIRI Score", f"{df['AIRI_composite'].mean():.1f}")
    col3.metric("Median AIRI Score", f"{df['AIRI_composite'].median():.1f}")
    col4.metric("Most Common Band", df['AIRI_band'].mode()[0] if not df['AIRI_band'].empty else "N/A")

    dim_cols = ['Strategy_Governance','Data_Technology','People_Skills','Risk_Ethics']
    available = [c for c in dim_cols if c in df.columns]
    if available:
        avg_dims = df[available].mean().to_dict()
        st.plotly_chart(create_radar_chart(avg_dims, "Average Dimension Profile (Cohort)"), use_container_width=True)
        st.plotly_chart(create_pca_scatter(df), use_container_width=True)

    col1,col2,col3 = st.columns(3)
    with col1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.page = 'results'; st.rerun()
    with col2:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()
    with col3:
        if st.button("🗑️ Clear My Data", use_container_width=True):
            clear_assessment(); st.session_state.page = 'home'; st.rerun()


def main():
    configure_page()
    init_session_state()

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:20px 0;">
            <h2 style="color:#1e3a5f;margin:0;">🤖 AIRI</h2>
            <p style="color:#718096;font-size:0.9em;">AI Readiness Index</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.page = 'home'; st.rerun()
        if st.button("📝 Take Assessment", use_container_width=True):
            st.session_state.page = 'survey'; st.rerun()
        if st.button("📊 My Results", use_container_width=True):
            if st.session_state.assessment_complete:
                st.session_state.page = 'results'; st.rerun()
            else:
                st.warning("Complete an assessment first!")
        if st.button("📈 Analytics Dashboard", use_container_width=True):
            st.session_state.page = 'dashboard'; st.rerun()
        st.markdown("---")
        st.markdown("""
        <div style="padding:15px;background:#ebf8ff;border-radius:8px;">
            <h4 style="color:#2c5282;margin:0 0 10px 0;">About AIRI</h4>
            <p style="color:#4a5568;font-size:0.85em;margin:0;">
                Quantitative diagnostic tool for UK debt management institutions to assess
                preparedness for ethical AI agent deployment. Aligned with FCA Consumer Duty.
            </p>
        </div>
        <br>
        <p style="color:#718096;font-size:0.8em;text-align:center;">
            Bournemouth University | MSc IT<br>COMP7024 | 2025/26<br>
            Validated by expert survey (n=121)
        </p>""", unsafe_allow_html=True)

    pages = {'home':render_home,'survey':render_survey,'results':render_results,'dashboard':render_dashboard}
    pages.get(st.session_state.page, render_home)()


if __name__ == "__main__":
    main()
