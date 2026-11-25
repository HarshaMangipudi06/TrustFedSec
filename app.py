import hashlib, html, json
from datetime import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from config import Config
from src.dataset import prepare
from src.coordinator import run

st.set_page_config(page_title='TrustFed-Sec', page_icon='·', layout='wide')

CSS = '''
<style>
:root {
  --ink: #dcebf0;
  --muted: #819fa9;
  --line: #1d3a45;
  --accent: #2ec4b6;
  --accent-dim: #1a6c66;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {display:none !important;}
[data-testid="stHeader"] {background:transparent !important;}
html, body, [class*="css"] {font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;}
h1,h2,h3 {color:var(--ink); letter-spacing:-0.025em; font-weight:600;}
.stButton button {border:1px solid var(--accent-dim); border-radius:6px; font-weight:600; letter-spacing:0.01em;}
.stButton button:hover {border-color:var(--accent);}
.brand {display:flex; align-items:center; gap:14px; margin:4px 0 22px;}
.mark {width:38px; height:38px; border:1px solid var(--accent); border-radius:7px; display:flex; align-items:center; justify-content:center; color:var(--accent); font-size:13px; font-weight:700; letter-spacing:0.06em;}
.brand-name {font-size:25px; font-weight:650; letter-spacing:-0.035em; color:var(--ink); line-height:1.1;}
.brand-sub {font-size:12px; color:var(--muted); margin-top:3px;}
.pill {margin-left:auto; border:1px solid var(--line); border-radius:999px; padding:5px 11px; font-size:10px; letter-spacing:0.09em; color:var(--muted); white-space:nowrap;}
.section {display:flex; align-items:center; gap:10px; margin:24px 0 10px; font-size:11px; font-weight:700; letter-spacing:0.14em; color:var(--muted); text-transform:uppercase;}
.section .num {color:var(--accent); font-size:10px;}
.section:after {content:""; flex:1; height:1px; background:var(--line);}
.note {border-left:2px solid var(--accent-dim); padding:3px 0 3px 14px; color:#a8c2ca; font-size:13px; line-height:1.55; margin-bottom:14px;}
.side-label {font-size:11px; font-weight:700; letter-spacing:0.14em; color:var(--muted); margin:6px 0 16px;}
.divider {height:1px; background:var(--line); margin:18px 0;}
.stat {border:1px solid var(--line); border-radius:8px; background:#0c2230; padding:12px 14px; min-height:64px;}
.stat-label {font-size:10px; letter-spacing:0.1em; color:var(--muted); text-transform:uppercase;}
.stat-value {font-size:20px; font-weight:600; color:var(--ink); margin-top:5px; line-height:1.15;}
.source-line {font-size:11px; color:var(--muted); margin-top:8px;}
.explain {border:1px solid var(--line); border-radius:8px; background:#0c2230; padding:14px 16px; margin-top:12px; font-size:13px; line-height:1.6; color:#b9d1d8;}
.explain strong {color:var(--accent);}
.foot {border-top:1px solid var(--line); margin-top:34px; padding-top:12px; font-size:11px; color:#5f7e89;}
[data-testid="stMetric"] {background:#0c2230; padding:14px; border-radius:8px; border:1px solid var(--line);}
[data-testid="stMetricLabel"] p {color:var(--muted);}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

def section(num, label):
    st.markdown(f'<div class="section"><span class="num">{num}</span><span>{label}</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="side-label">EXPERIMENT SETUP</div>', unsafe_allow_html=True)
    dataset = st.radio('Dataset', ['Synthetic demonstration', 'Local UNSW-NB15 CSV'])
    files = st.file_uploader('Upload one or more UNSW-NB15 CSV files', type=['csv'], accept_multiple_files=True) if dataset.startswith('Local') else []
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    partition = st.selectbox('Client partition', ['IID', 'Non-IID'])
    n = st.slider('Clients', 2, 10, 5)
    bad = st.slider('Compromised clients', 0, n, min(2,n))
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    attack = st.selectbox('Attack type', ['sign flip', 'label flip'])
    strength = st.slider('Sign-flip multiplier', 0.0, 15.0, 5.0, 0.5)
    fraction = st.slider('Label poisoning fraction', 0.0, 1.0, 0.5, 0.05)
    rounds = st.slider('Communication rounds', 1, 30, 10)
    seed = st.number_input('Random seed', min_value=0, max_value=1000000, value=42)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    with st.expander('Trust parameters'):
        beta = st.slider('History smoothing β', 0.0, 0.95, 0.6, 0.05)
        norm = st.slider('Log norm deviation threshold', 0.2, 3.0, 1.0, 0.1)
        cosine = st.slider('Cosine threshold', -0.5, 0.9, 0.2, 0.1)
        val = st.slider('Validation loss threshold', 0.005, 0.2, 0.02, 0.005, format='%.3f')
        floor = st.slider('Minimum trust weight', 0.01, 0.5, 0.1, 0.01)
    if st.button('Reset results', use_container_width=True):
        st.session_state.pop('results', None); st.rerun()

dataset_tag = 'SYNTHETIC DATA' if not dataset.startswith('Local') else 'USER-SUPPLIED CSV'
st.markdown(f'''<div class="brand"><div class="mark">TF</div><div><div class="brand-name">TrustFed-Sec</div><div class="brand-sub">Federated intrusion detection · local simulation</div></div><div class="pill">{dataset_tag}</div></div>''', unsafe_allow_html=True)
st.markdown('<div class="note">Five clients train the same logistic model. One or two send poisoned updates. The coordinator looks at each update, checks it against a clean validation set, and changes how much that client counts. This does not monitor real traffic.</div>', unsafe_allow_html=True)

cfg = Config(seed=int(seed),clients=n,compromised=bad,attack=attack,strength=strength,poison_fraction=fraction,rounds=rounds,partition=partition,beta=beta,norm_threshold=norm,cosine_threshold=cosine,validation_threshold=val,trust_floor=floor)
file_data = [(f.name, f.getvalue()) for f in files]
key = hashlib.sha256((repr(cfg)+dataset).encode()+b''.join(name.encode()+content for name,content in file_data)).hexdigest()
if 'results' not in st.session_state: st.session_state.results = {}
current = st.session_state.results.get(key, {})
if st.session_state.results and not current: st.warning('Settings or dataset changed. Previous results are hidden to prevent comparisons across configurations.')

section('01', 'Run scenarios')
st.caption('One run = one data split, one partition, one starting model. The three arms use the same compromised clients and the same attack.')
buttons = st.columns(4)
selected = None
for col, name in zip(buttons, ['Normal FedAvg','Poisoned FedAvg','TrustFed-Sec']):
    if col.button('Run '+name, use_container_width=True): selected=name
if buttons[3].button('Run all three (A/B/C)', type='primary', use_container_width=True): selected='ALL'
if selected:
    if dataset.startswith('Local') and not files: st.error('Upload at least one CSV containing the binary label column.')
    else:
        try:
            with st.spinner('Preparing data and running rounds…'):
                data=prepare(cfg, files if files else None)
                scenarios=[selected] if selected!='ALL' else ['Normal FedAvg','Poisoned FedAvg','TrustFed-Sec']
                progress=st.progress(0, text='Starting'); live=st.empty()
                for name in scenarios:
                    def update(r, rows, metrics, _name=name):
                        progress.progress(r/cfg.rounds, text=f'{_name} · round {r}/{cfg.rounds} · validation loss {metrics["loss"]:.4f} · accuracy {metrics["accuracy"]:.3f} · macro F1 {metrics["macro_f1"]:.3f}')
                        live.dataframe(pd.DataFrame(rows)[['client','status','trust','aggregation_weight','update_norm','cosine','validation_loss_delta']], hide_index=True, use_container_width=True)
                    result=run(data,cfg,name,update)
                    current[name]=result; st.session_state.results[key]=current
                    out=Path('results'); out.mkdir(exist_ok=True)
                    (out / f'{datetime.now():%Y%m%d_%H%M%S_%f}_{name.replace(" ","_")}.json').write_text(json.dumps(result,indent=2), encoding='utf-8')
                    progress.progress(1.0, text=f'{name} complete')
                live.empty()
                st.success(f'Completed and saved to results/: {", ".join(scenarios)}.')
        except Exception as e: st.error(f'Experiment could not run: {e}')

if current:
    first=next(iter(current.values()))
    short='Synthetic' if first['source'].startswith('Synthetic') else 'UNSW-NB15 (uploaded)'
    stats=[('Dataset',short),('Samples',f"{first['samples']:,}"),('Features',str(first['features'])),('Clients',str(n))]
    for col,(label,value) in zip(st.columns(4),stats):
        col.markdown(f'<div class="stat"><div class="stat-label">{label}</div><div class="stat-value">{html.escape(value)}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="source-line">{html.escape(first["source"])}</div>', unsafe_allow_html=True)

    section('02', 'Held-out test results')
    table=[]
    for name,result in current.items(): table.append({'Scenario':name, **result['test'], 'seconds':result['seconds'], 'communication_bytes':result['communication_bytes']})
    st.dataframe(pd.DataFrame(table).set_index('Scenario').round(4), use_container_width=True)
    st.caption('Attack recall is recall for label 1. Communication is the estimated float64 parameter bytes in both directions, excluding protocol overhead. Timing is from this machine.')

    hist=pd.concat([pd.DataFrame(v['history']).assign(Scenario=k) for k,v in current.items()],ignore_index=True)
    chart_cfg={'displayModeBar':False}
    a,b=st.columns(2)
    for col, metric in [(a,'accuracy'),(b,'macro_f1')]:
        fig=px.line(hist,x='round',y=metric,color='Scenario',markers=True,template='plotly_dark',title='Validation '+metric.replace('_',' '))
        fig.update_layout(height=300,margin=dict(l=8,r=8,t=44,b=8),font=dict(size=11))
        col.plotly_chart(fig,use_container_width=True,theme=None,config=chart_cfg)
    fig=px.line(hist,x='round',y='loss',color='Scenario',markers=True,template='plotly_dark',title='Validation loss')
    fig.update_layout(height=300,margin=dict(l=8,r=8,t=44,b=8),font=dict(size=11))
    st.plotly_chart(fig,use_container_width=True,theme=None,config=chart_cfg)

    section('03', 'Client inspection')
    c1,c2=st.columns([2,1])
    with c1: name=st.selectbox('Scenario',list(current),index=list(current).index('TrustFed-Sec') if 'TrustFed-Sec' in current else 0)
    with c2: round_no=st.select_slider('Round',options=list(range(1,cfg.rounds+1)),value=cfg.rounds)
    result=current[name]
    rows=pd.DataFrame(result['clients']); rows=rows[rows['round']==round_no]
    bars=rows.assign(kind=rows['simulated_compromised'].map({True:'simulated compromised',False:'honest'}))
    fig=px.bar(bars,x='client',y='trust',color='kind',range_y=[0,1],title=f'Trust by client · round {round_no}',template='plotly_dark',color_discrete_map={'simulated compromised':'#e4572e','honest':'#2ec4b6'})
    fig.update_layout(height=250,margin=dict(l=10,r=10,t=40,b=10),legend_title_text='',font=dict(size=11))
    fig.update_traces(width=0.55)
    st.plotly_chart(fig,use_container_width=True,theme=None,config=chart_cfg)
    shown=['client','status','trust','aggregation_weight','cosine','validation_loss_delta','update_norm','candidate_attack_recall','simulated_compromised']
    st.dataframe(rows[shown].round(4),hide_index=True,use_container_width=True,column_config={
        'trust':st.column_config.ProgressColumn('trust',format='%.3f',min_value=0.0,max_value=1.0),
        'aggregation_weight':st.column_config.ProgressColumn('weight',format='percent',min_value=0.0,max_value=1.0),
        'simulated_compromised':st.column_config.CheckboxColumn('simulated compromised',disabled=True),
    })
    client=st.selectbox('Explain client',rows['client'].tolist()); row=rows[rows.client==client].iloc[0]
    st.markdown(f'''<div class="explain">Client <strong>{client}</strong> carries trust <strong>{row['trust']:.3f}</strong> and gets <strong>{row['aggregation_weight']:.1%}</strong> of the update. Its update norm is {row['update_norm']:.3f}, its cosine to the median update is {row['cosine']:.3f}, and trying it on the validation set changed the loss by {row['validation_loss_delta']:+.4f}. Status: <strong>{row['status']}</strong>. A review flag is a signal, not proof of cheating.</div>''', unsafe_allow_html=True)
    if name != 'TrustFed-Sec': st.caption('In the FedAvg arms those trust values are diagnostic only; aggregation uses sample counts.')
    st.download_button('Download result JSON',json.dumps(result,indent=2),file_name='trustfed_result.json',mime='application/json')
else:
    st.markdown('<div class="note">No results yet. Run an arm above, or all three for a side-by-side comparison.</div>', unsafe_allow_html=True)

st.markdown('<div class="foot">TrustFed-Sec · local CPU simulation · results are written to results/</div>', unsafe_allow_html=True)
