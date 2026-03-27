"""
Network Anomaly Detection System — Streamlit App
Run:  streamlit run app/app.py
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
##pageconfiguration
st.set_page_config(
    page_title="Local Anomaly",
    layout="wide",
    initial_sidebar_state="expanded",
)
##cssforlocalanomaly
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Light Mode */
@media (prefers-color-scheme: light) {
    :root {
        --bg:           #f5f5f5;
        --bg-card:      #f0f0f0;
        --bg-card2:     #ebebeb;
        --cyan:         #0077aa;
        --green:        #00875a;
        --red:          #cc1f3f;
        --amber:        #b38600;
        --purple:       #6b2fa0;
        --text:         #1a1a2e;
        --muted:        #6b7280;
        --border:       #d1d5db;
    }
}

/* Dark Mode */
@media (prefers-color-scheme: dark) {
    :root {
        --bg:           #080c14;
        --bg-card:      #0d1321;
        --bg-card2:     #111c2e;
        --cyan:         #00c4e0;
        --green:        #00d484;
        --red:          #ff3860;
        --amber:        #f0b800;
        --purple:       #a855d4;
        --text:         #ccd6f6;
        --muted:        #4a5568;
        --border:       #172040;
    }
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', Times New Roman !important;
    color: var(--text) !important;
}
            
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Header ── */
.hero {
    background: linear-gradient(135deg, #f5f5f5 0%, #f0f0f0 60%, #ebebeb 100%);
    border: 1px solid #000000;
    border-radius: 18px;
    padding: 2rem 2.5rem 1.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent 0%, transparent 100%);
}
.hero::after {
    content:'';
    position:absolute; bottom:-60px; right:-60px;
    width:200px; height:200px;
    background: radial-gradient(circle, rgba(0,229,255,0.06) 0%, transparent 70%);
    border-radius:50%;
}
.hero h1 {
    font-family:'Times New Roman',monospace !important;
    font-size:2.1rem !important; font-weight:700 !important;
    color: var(--text) !important;
    letter-spacing:3px; margin:0 !important;
    text-shadow: 0 0 40px rgba(0,229,255,0.35);
}
.hero p { color:var(--muted) !important; margin:0.4rem 0 0 !important; font-size:0.82rem; letter-spacing:2px; }

/* ── Cards ── */
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.3rem 1.4rem 1rem;
    position: relative;
    overflow: hidden;
}
.kpi-card .val {
    font-family:'Times New Roman',monospace;
    font-size:1.9rem; font-weight:700; margin:0;
    line-height:1.1;
}
.kpi-card .lbl {
    font-size:0.7rem; letter-spacing:2px;
    text-transform:uppercase; color:var(--muted);
    margin-top:4px;
}
.kpi-card::after {
    content:''; position:absolute; bottom:0; left:0; right:0; height:3px;
    background: var(--accent, var(--text)); opacity:.7;
}

/* ── Result boxes ── */
.result-normal {
    background: linear-gradient(135deg,#04180f,#061a10);
    border: 2px solid var(--green);
    border-radius: 16px; padding:2rem 1.5rem;
    text-align:center;
    box-shadow: 0 0 50px rgba(0,255,157,0.12);
}
.result-attack {
    background: linear-gradient(135deg,#1a0510,#1f060d);
    border: 2px solid var(--red);
    border-radius: 16px; padding:2rem 1.5rem;
    text-align:center;
    box-shadow: 0 0 50px rgba(255,56,96,0.14);
}
.result-label {
    font-family:'IBM Plex Mono',monospace;
    font-size:1.5rem; font-weight:700; letter-spacing:3px;
}
.result-sub { font-size:0.82rem; color:var(--muted); margin-top:6px; }
.conf-bar-bg {
    background:rgba(255,255,255,0.06); border-radius:100px;
    height:7px; margin:1rem 0 0.3rem; overflow:hidden;
}

/* ── Section titles ── */
.sec {
    font-family:'Times New Roman',monospace;
    font-size:0.68rem; letter-spacing:3px;
    color: var(--text); text-transform:uppercase;
    margin-bottom:1rem; padding-bottom:0.5rem;
    border-bottom:1px solid var(--border);
}

/* ── Buttons ── */
.stButton>button {
    background:linear-gradient(135deg,#2a7ae0, #1a55b0) !important;
    color:#fff !important; border:1px solid #1464c8 !important;
    border-radius:8px !important;
    font-family:'Times New Roman',monospace !important;
    font-weight:600 !important; letter-spacing:1px !important;
    padding:0.55rem 1.4rem !important;
    transition:all .2s !important;
}
.stButton>button:hover {
    background:linear-gradient(135deg,#2a7ae0, #1a55b0) !important;
    box-shadow:0 0 22px rgba(20,100,200,0.45) !important;
    transform:translateY(-1px) !important;
}

/* ── Misc ── */
.stSelectbox [data-baseweb="select"] {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
}
.stTabs [data-baseweb="tab-list"] { background:var(--bg-card) !important; border-radius:8px !important; }
.stTabs [data-baseweb="tab"]      { color:var(--muted) !important; }
.stTabs [aria-selected="true"]    { color:var(--text) !important; }
div[data-testid="stMarkdownContainer"] p { color:var(--text) !important; }
</style>
""", unsafe_allow_html=True)


##Loaders
@st.cache_resource
def load_models():
    def pkl(name):
        p = os.path.join(MODELS_DIR, name)
        if not os.path.exists(p): return None
        with open(p, "rb") as f: return pickle.load(f)
    return {
        "preprocessor":       pkl("preprocessor.pkl"),
        "rf":                 pkl("rf_model.pkl"),
        "gb":                 pkl("gb_model.pkl"),
        "iso":                pkl("iso_forest.pkl"),
        "feature_names":      pkl("feature_names.pkl"),
        "label_encoder":      pkl("label_encoder.pkl"),
        "feature_importance": pkl("feature_importance.pkl"),
    }

@st.cache_data
def load_test():
    p = os.path.join(DATA_DIR, "test.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_metrics():
    p = os.path.join(MODELS_DIR, "metrics.json")
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f)

CAT_COLS = ["protocol_type", "service", "flag"]

def predict(models, sample_df, key):
    pre = models["preprocessor"]
    if pre is None: return None, None
    X = pre.transform(sample_df)
    m = models[key]
    if m is None: return None, None
    if key == "iso":
        raw  = m.predict(X)[0]
        pred = 1 if raw == -1 else 0
        prob = float(np.clip(-m.score_samples(X)[0] / 0.5, 0, 1))
    else:
        pred = int(m.predict(X)[0])
        prob = float(m.predict_proba(X)[0, 1])
    return pred, prob

def theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#ccd6f6"),
        xaxis=dict(gridcolor="#172040", zerolinecolor="#172040"),
        yaxis=dict(gridcolor="#172040", zerolinecolor="#172040"),
    )


##sidebar
with st.sidebar:
    st.markdown("### Local Anomaly")
    st.markdown("---")
    page = st.radio("Navigate", ["Anomaly Detector", "Model Performance", "EDA & Insights"])
    st.markdown("---")
    st.markdown("**Active Model**")
    model_choice = st.selectbox("", ["Random Forest", "Gradient Boosting", "Isolation Forest"])
    model_key = {"Random Forest":"rf", "Gradient Boosting":"gb", "Isolation Forest":"iso"}[model_choice]
    st.markdown("---")
    st.markdown("**Dataset**")
    st.caption("NSL-KDD — Canadian Institute for Cybersecurity")
    st.caption("41 features · binary + multi-class labels")
    st.markdown("---")
    st.markdown("**Attack Categories**")
    st.markdown("🔴 DoS &nbsp;&nbsp; 🟠 Probe &nbsp;&nbsp; 🟡 R2L")
    st.markdown("🟤 U2R &nbsp;&nbsp; 🟢 Normal")

##loadeditems
models  = load_models()
test_df = load_test()
metrics = load_metrics()

models_ready = models["rf"] is not None
data_ready   = test_df is not None
feat_cols    = [c for c in (test_df.columns if data_ready else []) if c not in ["binary_label","attack_type"]]


##pg-1 Anamolay Detector
if "Detector" in page:

    st.markdown("""
    <div class="hero">
        <h1>Local Anomaly</h1>
        <p>REAL-TIME NETWORK ANOMALY DETECTION · NSL-KDD BENCHMARK · ML-POWERED</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready or not models_ready:
        st.error("Data or models not found. Run the setup scripts first:")
        st.code("python scripts/download_data.py\npython scripts/train_model.py", language="bash")
        st.stop()

    ##sampleselector
    st.markdown('<div class="sec">Sample Selection</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2.5, 2.5, 1])
    with c1:
        mode = st.selectbox("Selection Mode", ["Random Sample", "Filter by Attack Type", "By Index"])
    with c2:
        if mode == "Filter by Attack Type":
            atype = st.selectbox("Attack Type", sorted(test_df["attack_type"].unique()))
        elif mode == "By Index":
            idx_input = st.number_input("Row Index", 0, len(test_df)-1, 0)
        else:
            atype = None
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        pick = st.button("Pick Sample")

    if "sidx" not in st.session_state: st.session_state.sidx = 0

    if pick or "srow" not in st.session_state:
        if mode == "Random Sample":
            st.session_state.sidx = int(np.random.randint(0, len(test_df)))
        elif mode == "Filter by Attack Type":
            sub = test_df[test_df["attack_type"] == atype]
            st.session_state.sidx = int(sub.sample(1).index[0])
        else:
            st.session_state.sidx = int(idx_input)
        st.session_state.srow = test_df.iloc[st.session_state.sidx]

    sample     = st.session_state.srow
    sample_df  = pd.DataFrame([sample[feat_cols]])
    true_label = int(sample["binary_label"])
    true_type  = sample["attack_type"]

    pred, prob = predict(models, sample_df, model_key)
    if pred is None:
        st.warning("Model or preprocessor not loaded."); st.stop()

    correct = pred == true_label

    ###result&width
    st.markdown('<div class="sec" style="margin-top:1.4rem;">Detection Result</div>', unsafe_allow_html=True)
    col_res, col_gauge = st.columns([3, 2])

    with col_res:
        if pred == 0:
            conf_bar = f'<div class="conf-bar-bg"><div style="width:{(1-prob)*100:.0f}%;height:100%;background:var(--green);border-radius:100px;"></div></div>'
            st.markdown(f"""
            <div class="result-normal">
                <div style="font-size:3rem;"></div>
                <div class="result-label" style="color:#00ff9d;">NORMAL TRAFFIC</div>
                <div class="result-sub">No anomaly detected &nbsp;·&nbsp; Confidence {(1-prob)*100:.1f}%</div>
                {conf_bar}
            </div>""", unsafe_allow_html=True)
        else:
            conf_bar = f'<div class="conf-bar-bg"><div style="width:{prob*100:.0f}%;height:100%;background:var(--red);border-radius:100px;"></div></div>'
            st.markdown(f"""
            <div class="result-attack">
                <div style="font-size:3rem;"></div>
                <div class="result-label" style="color:#ff3860;">ANOMALY DETECTED</div>
                <div class="result-sub">Potential network intrusion &nbsp;·&nbsp; Threat score {prob*100:.1f}%</div>
                {conf_bar}
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Prediction",   "Attack" if pred else "Normal")
        m2.metric("Ground Truth", "Attack" if true_label else "Normal",
                  delta="✓ Correct" if correct else "✗ Wrong",
                  delta_color="normal" if correct else "inverse")
        m3.metric("Attack Type",  true_type.upper())
        m4.metric("Sample Index", st.session_state.sidx)

    with col_gauge:
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix":"%","font":{"size":30,"family":"IBM Plex Mono","color":"#ccd6f6"}},
            title={"text":"Threat Score","font":{"size":13,"color":"#4a5568"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#4a5568","tickfont":{"size":10}},
                "bar":{"color":"#ff3860" if pred else "#00ff9d","thickness":0.25},
                "bgcolor":"#0d1321","borderwidth":0,
                "steps":[
                    {"range":[0, 40], "color":"#061a10"},
                    {"range":[40,70], "color":"#1a1506"},
                    {"range":[70,100],"color":"#1a0510"},
                ],
                "threshold":{"line":{"color":"#00e5ff","width":2},"value":50},
            },
        ))
        fig_g.update_layout(height=270, margin=dict(t=40,b=10,l=20,r=20), **theme())
        st.plotly_chart(fig_g, use_container_width=True)

    ###featuredtable
    st.markdown('<div class="sec" style="margin-top:1.6rem;">Feature Breakdown</div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs([" Key Features", " All Features"])

    KEY = {
        "duration":"Connection duration (sec)",
        "protocol_type":"Network protocol",
        "service":"Target service",
        "flag":"Connection status flag",
        "src_bytes":"Bytes from source → dest",
        "dst_bytes":"Bytes from dest → source",
        "land":"Same src/dst IP:port",
        "logged_in":"Login status (1=yes)",
        "count":"Connections same host / 2 sec",
        "srv_count":"Connections same service / 2 sec",
        "serror_rate":"% SYN error connections",
        "rerror_rate":"% REJ error connections",
        "same_srv_rate":"% connections same service",
        "diff_srv_rate":"% connections diff service",
        "dst_host_count":"Dest host connection count",
        "hot":"Hot indicators count",
        "num_compromised":"# compromised conditions",
        "num_failed_logins":"# failed login attempts",
    }
    with tab1:
        rows = [{"Feature":k,"Description":v,"Value":str(sample_df[k].values[0])}
                for k,v in KEY.items() if k in sample_df.columns]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={
                         "Feature":st.column_config.TextColumn(width=170),
                         "Description":st.column_config.TextColumn(width=260),
                         "Value":st.column_config.TextColumn(width=130),
                     })
    with tab2:
        all_rows = [{"Feature":c,"Value":str(sample_df[c].values[0])} for c in feat_cols]
        st.dataframe(pd.DataFrame(all_rows), use_container_width=True, hide_index=True)

    #chartforradar
    st.markdown('<div class="sec" style="margin-top:1.6rem;">Traffic Radar Profile</div>', unsafe_allow_html=True)
    radar_f = ["duration","src_bytes","dst_bytes","count","srv_count",
               "serror_rate","rerror_rate","same_srv_rate","dst_host_count","hot"]
    ranges  = [1000, 1e7, 1e7, 512, 512, 1, 1, 1, 256, 30]
    rvals   = [min(float(sample_df[f].values[0] if f in sample_df.columns else 0)/r, 1.0)
               for f,r in zip(radar_f,ranges)]

    color_fill = "rgba(255,56,96,0.15)" if pred else "rgba(0,255,157,0.15)"
    color_line = "#ff3860" if pred else "#00ff9d"

    fig_r = go.Figure(go.Scatterpolar(
        r=rvals+[rvals[0]], theta=radar_f+[radar_f[0]],
        fill="toself", fillcolor=color_fill,
        line=dict(color=color_line, width=2),
    ))
    fig_r.update_layout(
        polar=dict(
            bgcolor="#0d1321",
            radialaxis=dict(visible=True,range=[0,1],color="#4a5568",gridcolor="#172040",tickfont=dict(size=9)),
            angularaxis=dict(color="#4a5568",gridcolor="#172040",tickfont=dict(size=10)),
        ),
        height=390, margin=dict(t=30,b=30,l=50,r=50), showlegend=False,
        **theme(),
    )
    c_rad, c_bar = st.columns([3,2])
    with c_rad:
        st.plotly_chart(fig_r, use_container_width=True)

    ###featureimportance
    with c_bar:
        fi = models.get("feature_importance")
        if fi:
            fi_df = pd.DataFrame(list(fi.items()), columns=["feat","imp"]).head(12)
            # intersect with sample features
            sample_arr = models["preprocessor"].transform(sample_df)[0]
            fn = models["feature_names"] or []
            contrib = {}
            for feat, imp in fi_df.values:
                if feat in fn:
                    idx = fn.index(feat)
                    contrib[feat] = imp * abs(float(sample_arr[idx]))
            if contrib:
                cb_df = pd.DataFrame(list(contrib.items()),columns=["Feature","Score"]).sort_values("Score",ascending=True).tail(10)
                fig_cb = go.Figure(go.Bar(
                    x=cb_df["Score"], y=cb_df["Feature"], orientation="h",
                    marker=dict(color=cb_df["Score"],colorscale=[[0,"#0d1321"],[0.5,"#0a3d80"],[1,"#00e5ff"]]),
                ))
                fig_cb.update_layout(title="Feature Contribution",height=390,
                                      margin=dict(t=40,b=20,l=10,r=10), **theme())
                st.plotly_chart(fig_cb, use_container_width=True)

###pg2modelperformance
elif "Performance" in page:

    st.markdown("""
    <div class="hero">
        <h1>MODEL PERFORMANCE</h1>
        <p>EVALUATION METRICS · CONFUSION MATRICES · FEATURE IMPORTANCE</p>
    </div>
    """, unsafe_allow_html=True)

    if not metrics:
        st.error("No metrics found. Train the models: `python scripts/train_model.py`"); st.stop()

    MLABELS = {
        "random_forest":    "Random Forest",
        "gradient_boosting":"Gradient Boosting",
        "isolation_forest": "Isolation Forest",
    }

    ###kpicards
    st.markdown('<div class="sec">Model Comparison</div>', unsafe_allow_html=True)
    cols = st.columns(len(metrics))
    ACCENTS = ["#00e5ff","#00ff9d","#ff3860"]
    for col,(k,v),acc in zip(cols, metrics.items(), ACCENTS):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent:{acc}">
                <p class="lbl">{MLABELS.get(k,k)}</p>
                <p class="val" style="color:{acc}">{v['accuracy']*100:.2f}%</p>
                <p class="lbl">ACCURACY</p>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            for mk in ["f1","precision","recall"]:
                if v.get(mk): st.metric(mk.upper(), f"{v[mk]:.4f}")

    ###groupedbar
    st.markdown('<div class="sec" style="margin-top:2rem;">Metrics Comparison</div>', unsafe_allow_html=True)
    rows=[]
    for k,v in metrics.items():
        for mk in ["accuracy","f1","precision","recall"]:
            rows.append({"Model":MLABELS.get(k,k),"Metric":mk.upper(),"Score":v.get(mk,0)})
    fig_b = px.bar(pd.DataFrame(rows), x="Model", y="Score", color="Metric", barmode="group",
                   color_discrete_sequence=["#00e5ff","#00ff9d","#ffcc00","#ff3860"])
    fig_b.update_layout(height=400, yaxis=dict(range=[0.65,1.01]), **theme())
    st.plotly_chart(fig_b, use_container_width=True)

    ###confusionmatrix
    st.markdown('<div class="sec">Confusion Matrices</div>', unsafe_allow_html=True)
    cm_cols = st.columns(len(metrics))
    for col,(k,v) in zip(cm_cols, metrics.items()):
        cm = np.array(v["confusion_matrix"])
        fig_cm = px.imshow(cm, text_auto=True,
                           labels=dict(x="Predicted",y="Actual"),
                           x=["Normal","Attack"], y=["Normal","Attack"],
                           color_continuous_scale=[[0,"#080c14"],[0.5,"#0a3d80"],[1,"#00e5ff"]],
                           title=MLABELS.get(k,k))
        fig_cm.update_layout(height=310, **theme())
        col.plotly_chart(fig_cm, use_container_width=True)

    ###featureimportance
    fi = models.get("feature_importance")
    if fi:
        st.markdown('<div class="sec">Top 20 Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
        fi_df = pd.DataFrame(list(fi.items()),columns=["Feature","Importance"]).head(20)
        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance",
                        color_continuous_scale=[[0,"#0d1321"],[0.4,"#0a3d80"],[1,"#00e5ff"]])
        fig_fi.update_layout(height=560, yaxis=dict(autorange="reversed"), **theme())
        st.plotly_chart(fig_fi, use_container_width=True)

    #rocaucsummary
    auc_vals = {MLABELS.get(k,k): v["auc"] for k,v in metrics.items() if v.get("auc")}
    if auc_vals:
        st.markdown('<div class="sec">ROC-AUC Summary</div>', unsafe_allow_html=True)
        acols = st.columns(len(auc_vals))
        for col,(name,auc) in zip(acols, auc_vals.items()):
            col.metric(name, f"{auc:.4f}")


###edainsights
elif "EDA" in page:

    st.markdown("""
    <div class="hero">
        <h1>DATA INSIGHTS</h1>
        <p>EXPLORATORY DATA ANALYSIS · NSL-KDD TEST SET · TRAFFIC PATTERNS</p>
    </div>
    """, unsafe_allow_html=True)

    if not data_ready:
        st.error("Test data not found. Run `python scripts/download_data.py` first."); st.stop()

    ###overview
    st.markdown('<div class="sec">Dataset Overview</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Total Records", f"{len(test_df):,}")
    k2.metric("Features",      len(feat_cols))
    k3.metric("Normal",        f"{(test_df['binary_label']==0).sum():,}")
    k4.metric("Attacks",       f"{(test_df['binary_label']==1).sum():,}")

    ###attacktypesanddistance
    st.markdown('<div class="sec" style="margin-top:1.4rem;">Traffic Composition</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        at = test_df["attack_type"].value_counts().reset_index()
        at.columns = ["Type","Count"]
        fig_pie = px.pie(at, values="Count", names="Type", hole=0.45,
                         color_discrete_sequence=["#00e5ff","#ff3860","#00ff9d","#ffcc00","#9d4edd",
                                                   "#ff7f50","#4169e1","#00ced1","#ff69b4"])
        fig_pie.update_layout(height=360, **theme())
        fig_pie.update_traces(textfont_size=11)
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        p_df = test_df.groupby(["protocol_type","binary_label"]).size().reset_index(name="n")
        p_df["Label"] = p_df["binary_label"].map({0:"Normal",1:"Attack"})
        fig_proto = px.bar(p_df, x="protocol_type", y="n", color="Label", barmode="group",
                           color_discrete_map={"Normal":"#00ff9d","Attack":"#ff3860"})
        fig_proto.update_layout(height=360, xaxis_title="Protocol", yaxis_title="Count", **theme())
        st.plotly_chart(fig_proto, use_container_width=True)

    ###bytedistribution
    st.markdown('<div class="sec">Byte Transfer Distribution (log scale)</div>', unsafe_allow_html=True)
    fig_by = go.Figure()
    for label, color, name in [(0,"#00ff9d","Normal"),(1,"#ff3860","Attack")]:
        vals = test_df[test_df["binary_label"]==label]["src_bytes"].clip(1)
        fig_by.add_trace(go.Histogram(x=np.log1p(vals), name=name, opacity=0.72,
                                       marker_color=color, nbinsx=60))
    fig_by.update_layout(barmode="overlay", height=340,
                          xaxis_title="log(src_bytes+1)", yaxis_title="Count",
                          legend=dict(bgcolor="rgba(0,0,0,0)"), **theme())
    st.plotly_chart(fig_by, use_container_width=True)

    ###Scattererrorratevscount
    st.markdown('<div class="sec">Connection Count vs SYN Error Rate</div>', unsafe_allow_html=True)
    sdf = test_df.sample(min(3000, len(test_df)), random_state=42).copy()
    sdf["Label"] = sdf["binary_label"].map({0:"Normal",1:"Attack"})
    fig_sc = px.scatter(sdf, x="count", y="serror_rate", color="Label",
                         opacity=0.55,
                         color_discrete_map={"Normal":"#00ff9d","Attack":"#ff3860"})
    fig_sc.update_layout(height=380, xaxis_title="Connection Count",
                          yaxis_title="SYN Error Rate", **theme())
    st.plotly_chart(fig_sc, use_container_width=True)

    ###Serviceheatmap
    st.markdown('<div class="sec">Top Services by Attack Frequency</div>', unsafe_allow_html=True)
    svc_df = test_df.groupby(["service","binary_label"]).size().unstack(fill_value=0)
    svc_df.columns = ["Normal","Attack"]
    svc_df["Attack_Rate"] = svc_df["Attack"] / (svc_df["Normal"]+svc_df["Attack"]+1e-9)
    top_svc = svc_df.sort_values("Attack", ascending=False).head(15).reset_index()
    fig_svc = go.Figure(go.Bar(
        x=top_svc["service"], y=top_svc["Attack"],
        marker=dict(color=top_svc["Attack_Rate"],
                    colorscale=[[0,"#0d1321"],[0.4,"#4a0080"],[1,"#ff3860"]],
                    colorbar=dict(title="Attack Rate")),
    ))
    fig_svc.update_layout(height=360, xaxis_title="Service", yaxis_title="Attack Count", **theme())
    st.plotly_chart(fig_svc, use_container_width=True)
