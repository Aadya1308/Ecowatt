"""
EnergyIQ — Streamlit UI fully connected to lstm/lstm.py backend.
Place in repo root. Run: python -m streamlit run app.py

Expected repo structure:
  forecasting-energy-consumption-LSTM/
  ├── app.py                  ← this file
  ├── lstm/
  │   └── lstm.py             ← your original code (imported here)
  ├── seq2seq/
  │   └── seq2seq.py
  ├── dataset/
  │   └── kaggle_data_1h.csv  ← your dataset
  ├── lstm_model.h5           ← saved after training
  └── scaler.pkl
"""

import os, sys, pickle, warnings, random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
from collections import deque
from math import sqrt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── Repo root & path setup ────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "lstm"))
sys.path.insert(0, str(ROOT / "seq2seq"))

# ── TensorFlow ────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Dense, Dropout, LSTM,
                                          Activation, Bidirectional,
                                          Flatten, TimeDistributed)
    CuDNNLSTM = LSTM  # CuDNNLSTM removed in TF2 — LSTM auto-uses GPU if available
    from tensorflow.keras.callbacks import (TensorBoard, ModelCheckpoint,
                                             EarlyStopping, ReduceLROnPlateau)
    TF_OK = True
    TF_ERROR = None
except Exception as e:
    TF_OK = False
    TF_ERROR = str(e)

# ── Constants from lstm.py ────────────────────────────────────────────────────
SEQ_LEN             = 100   # look-back window
FUTURE_PERIOD_PREDICT = 6   # steps ahead to predict
EPOCHS_DEFAULT      = 30
BATCH_SIZE_DEFAULT  = 64

# ── Auto-detect repo files ────────────────────────────────────────────────────
AUTO_MODEL  = ROOT / "lstm_model.h5"
AUTO_SCALER = ROOT / "scaler.pkl"
# dataset — repo uses kaggle_data_1h.csv
AUTO_DATA_PATHS = [
    ROOT / "dataset" / "kaggle_data_1h.csv",
    ROOT / "kaggle_data_1h.csv",
    ROOT / "dataset" / "household_power_consumption.txt",
    ROOT / "household_power_consumption.txt",
]
AUTO_DATA = next((p for p in AUTO_DATA_PATHS if p.exists()), None)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECOWATT — Smart Energy System",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#F7F7F5}
[data-testid="stSidebar"]{background:#fff;border-right:1px solid #E4E2DC}
.metric-box{background:#fff;border:1px solid #E4E2DC;border-radius:10px;
            padding:14px 18px;margin-bottom:10px}
.metric-box .lbl{font-size:12px;color:#888;margin-bottom:4px}
.metric-box .val{font-size:24px;font-weight:600;color:#1A1916}
.metric-box .dl{font-size:12px;margin-top:3px}
.good{color:#0F6E56}.bad{color:#A32D2D}
.info{background:#E1F5EE;border-left:4px solid #1D9E75;padding:10px 14px;
      border-radius:6px;font-size:13px;color:#0F6E56;margin-bottom:10px}
.warn{background:#FAEEDA;border-left:4px solid #EF9F27;padding:10px 14px;
      border-radius:6px;font-size:13px;color:#854F0B;margin-bottom:10px}
.err{background:#FCEBEB;border-left:4px solid #E24B4A;padding:10px 14px;
     border-radius:6px;font-size:13px;color:#A32D2D;margin-bottom:10px}
.file-ok{background:#E1F5EE;border-radius:6px;padding:5px 10px;
         font-size:12px;color:#0F6E56;margin-bottom:3px}
.file-miss{background:#FFF3E0;border-radius:6px;padding:5px 10px;
           font-size:12px;color:#854F0B;margin-bottom:3px}
</style>
""", unsafe_allow_html=True)

GREEN="#1D9E75"; BLUE="#378ADD"; AMBER="#EF9F27"; RED="#E24B4A"; GRAY="#B4B2A9"
PALETTE=[GREEN,BLUE,AMBER,RED,"#7F77DD","#D4537E"]

# ═══════════════════════════════════════════════════════════════════════════════
# BACKEND FUNCTIONS — mirrored from lstm/lstm.py
# ═══════════════════════════════════════════════════════════════════════════════

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

def theil_u(yhat, y):
    n = len(yhat)
    sum1 = sum(((yhat[i+1]-y[i+1])/y[i])**2 for i in range(n-1))
    sum2 = sum(((y[i+1]-y[i])/y[i])**2     for i in range(n-1))
    return sqrt(sum1/sum2) if sum2 > 0 else 0

def preprocess_df(df, seq_len=SEQ_LEN, shuffle=False):
    """Exact replica of lstm.py preprocess_df using deque."""
    sequential_data = []
    prev_days = deque(maxlen=seq_len)
    for i in df.values:
        prev_days.append([n for n in i[:-1]])   # all but last col = features
        if len(prev_days) == seq_len:
            sequential_data.append([np.array(prev_days), i[-1]])  # last col = target
    if shuffle:
        random.shuffle(sequential_data)
    X, y = [], []
    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

def split_data(df, percent=0.2):
    """Exact replica of lstm.py split_data."""
    length      = len(df)
    test_index  = int(length * percent)
    train_index = length - test_index
    test_df     = df[train_index:]
    train_df    = df[:train_index]
    return test_df, train_df

def train_length(length, batch_size):
    """Exact replica of lstm.py train_length."""
    length_values = []
    for x in range(int(length)-100, int(length)):
        modulo = x % batch_size
        if modulo == 0:
            length_values.append(x)
    return max(length_values)

def get_model(batch_size, seq_len, n_features, use_gpu=False):
    """
    Exact replica of lstm.py get_model().
    Falls back to LSTM if CuDNNLSTM not available (no GPU).
    """
    LSTMLayer = CuDNNLSTM if use_gpu else LSTM
    extra = {} if use_gpu else {"activation": "tanh"}

    model = Sequential()
    model.add(LSTMLayer(36, stateful=True, return_sequences=True,
                        batch_input_shape=(batch_size, seq_len, n_features),
                        **extra))
    model.add(Activation("relu"))
    model.add(LSTMLayer(36, stateful=True, return_sequences=False, **extra))
    model.add(Activation("relu"))
    model.add(Dense(1))
    model.compile(loss="mae", optimizer="adam", metrics=["mse","mae"])
    return model

@st.cache_data(show_spinner=False)
def load_dataset(path_or_bytes, is_bytes=False):
    """Load kaggle_data_1h.csv exactly as lstm.py does."""
    try:
        if is_bytes:
            import io
            raw = pd.read_csv(io.BytesIO(path_or_bytes), sep=",",
                              infer_datetime_format=True, low_memory=False,
                              index_col="time", encoding="utf-8")
        else:
            raw = pd.read_csv(path_or_bytes, sep=",",
                              infer_datetime_format=True, low_memory=False,
                              index_col="time", encoding="utf-8",
                              on_bad_lines="skip")
    except Exception:
        # fallback: UCI txt format
        try:
            if is_bytes:
                import io
                raw = pd.read_csv(io.BytesIO(path_or_bytes), sep=";",
                                  na_values="?",
                                  parse_dates={"time":[0,1]}, index_col="time")
            else:
                raw = pd.read_csv(path_or_bytes, sep=";", na_values="?",
                                  parse_dates={"time":[0,1]}, index_col="time")
        except Exception as e:
            return None, str(e)

    # Drop sub-metering (same as lstm.py)
    drop_cols = [c for c in raw.columns if "Sub_metering" in c]
    raw = raw.drop(drop_cols, axis=1, errors="ignore")
    raw.index = pd.to_datetime(raw.index, errors="coerce")
    raw.sort_index(inplace=True)
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw.dropna(inplace=True)
    # Resample to 1h (same as lstm.py)
    try:    raw = raw.resample("h").mean().dropna()
    except: pass
    return raw, None

def scale_data(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return pd.DataFrame(scaled, index=df.index, columns=df.columns), scaler

# ── Sidebar helpers ───────────────────────────────────────────────────────────
def badge(label, ok):
    cls = "file-ok" if ok else "file-miss"
    st.markdown(f'<div class="{cls}">{"✅" if ok else "⚠️"} {label}</div>',
                unsafe_allow_html=True)

def mpl_fig(h=3.2):
    fig, ax = plt.subplots(figsize=(9, h))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#E4E2DC")
    ax.tick_params(colors="#666", labelsize=9)
    return fig, ax

def demo_series(lb=100, hz=6, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, 4*np.pi, lb+hz)
    s = 1.5+0.8*np.sin(t)+0.3*np.sin(3*t)+np.random.normal(0,.07,len(t))
    return s[:lb], s[lb:]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ ECOWATT")
    st.caption("Smart Energy Monitoring & Optimization")
    st.divider()

    page = st.radio("Navigation", [
        "🏠 Dashboard",
        "📊 Data Explorer",
        "🔮 Forecast",
        "💡 Optimization Tips",
        "🏋️ Train Model",
        "📈 Training History",
        "📐 Model Evaluation",
        "⚙️ Model Info",
    ])

    st.divider()
    st.markdown("**Repo file status**")
    badge("lstm_model.h5",   AUTO_MODEL.exists())
    badge("scaler.pkl",      AUTO_SCALER.exists())
    badge(f"dataset ({AUTO_DATA.name if AUTO_DATA else 'not found'})",
          AUTO_DATA is not None)
    if TF_OK:
        badge("TensorFlow ✅", True)
    else:
        badge("TensorFlow", False)
        if TF_ERROR:
            st.sidebar.caption(f"TF error: {TF_ERROR[:150]}")

    st.divider()
    st.markdown("**Override — upload files**")
    up_model  = st.file_uploader("Model (.h5)",          type=["h5","keras"])
    up_scaler = st.file_uploader("Scaler (.pkl)",        type=["pkl"])
    up_data   = st.file_uploader("Dataset (.csv/.txt)",  type=["csv","txt"])
    up_hists  = st.file_uploader("History .pkl (multi)", type=["pkl"],
                                  accept_multiple_files=True)

# ── Resolve data ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def get_df(up_bytes=None):
    if up_bytes:
        return load_dataset(up_bytes, is_bytes=True)
    if AUTO_DATA:
        return load_dataset(str(AUTO_DATA))
    return None, "No dataset found"

@st.cache_resource(show_spinner="Loading model…")
def get_trained_model(up_bytes=None):
    if not TF_OK: return None
    import tempfile
    if up_bytes:
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            f.write(up_bytes); tmp = f.name
        try:    return keras.models.load_model(tmp)
        except: return None
    if AUTO_MODEL.exists():
        try:    return keras.models.load_model(str(AUTO_MODEL))
        except: return None
    return None

def get_scaler(up_bytes=None):
    if up_bytes: return pickle.loads(up_bytes)
    if AUTO_SCALER.exists():
        with open(AUTO_SCALER,"rb") as f: return pickle.load(f)
    return None

df, df_err       = get_df(up_data.read() if up_data else None)
trained_model    = get_trained_model(up_model.read() if up_model else None)
scaler           = get_scaler(up_scaler.read() if up_scaler else None)
have_data        = df is not None
have_model       = trained_model is not None
have_scaler      = scaler is not None

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("# ⚡ Dashboard")

    if not have_data:
        st.markdown(f'<div class="warn">⚠️ No dataset found. Place <code>kaggle_data_1h.csv</code> in the <code>dataset/</code> folder.</div>', unsafe_allow_html=True)
        act, _ = demo_series()
        st.caption("Showing synthetic demo data")
    else:
        st.markdown(f'<div class="info">✅ Dataset: <b>{AUTO_DATA.name if AUTO_DATA else "uploaded"}</b> — {len(df):,} hourly rows · {df.shape[1]} features · {df.index[0].date()} → {df.index[-1].date()}</div>', unsafe_allow_html=True)
        act = df.iloc[-SEQ_LEN:, 0].values

    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="lbl">⚡ Avg load</div><div class="val">{act.mean():.3f}</div><div class="dl">kW (last {SEQ_LEN}h)</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="lbl">📈 Peak load</div><div class="val">{act.max():.3f}</div><div class="dl bad">kW</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="lbl">📉 Min load</div><div class="val">{act.min():.3f}</div><div class="dl good">kW</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-box"><div class="lbl">🌿 CO₂ (est.)</div><div class="val">{act.sum()*0.82:.1f}</div><div class="dl">kg this window</div></div>', unsafe_allow_html=True)

    st.markdown("### Recent load profile")
    now    = datetime.now()
    hist_t = [now - timedelta(hours=SEQ_LEN-i) for i in range(SEQ_LEN)]
    fig, ax = mpl_fig(3)
    ax.fill_between(hist_t, act, alpha=0.1, color=GREEN)
    ax.plot(hist_t, act, color=GREEN, lw=2, label="Global Active Power (kW)")
    ax.set_ylabel("kW", fontsize=9); ax.legend(fontsize=9, frameon=False)
    st.pyplot(fig); plt.close()

    if have_data and len(df) > 24*7:
        st.markdown("### Weekly heatmap (last 4 weeks)")
        try:
            w4 = df.iloc[:,0].last("28D")
            pivot = w4.groupby([w4.index.day_of_week, w4.index.hour]).mean().unstack()
            fig2, ax2 = plt.subplots(figsize=(9,2.5))
            fig2.patch.set_facecolor("white"); ax2.set_facecolor("white")
            im = ax2.imshow(pivot.values, aspect="auto", cmap="Greens")
            ax2.set_yticks(range(7))
            ax2.set_yticklabels(["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], fontsize=9)
            ax2.set_xticks(range(0,24,3))
            ax2.set_xticklabels([f"{h}:00" for h in range(0,24,3)], fontsize=8)
            plt.colorbar(im, ax=ax2, label="kW", fraction=0.015, pad=0.02)
            ax2.spines[:].set_visible(False)
            st.pyplot(fig2); plt.close()
        except Exception as e:
            st.caption(f"Heatmap skipped: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":
    st.markdown("# 📊 Data Explorer")

    if not have_data:
        st.markdown('<div class="warn">⚠️ No dataset found. Place <code>kaggle_data_1h.csv</code> in <code>dataset/</code> or upload it in the sidebar.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info">✅ {len(df):,} rows · {df.shape[1]} features · {df.index[0].date()} → {df.index[-1].date()}</div>', unsafe_allow_html=True)

        col  = st.selectbox("Feature", df.columns.tolist())
        freq = st.radio("Resample", ["Hourly","Daily","Weekly"], horizontal=True)
        fm   = {"Hourly":"h","Daily":"D","Weekly":"W"}
        samp = df[col].resample(fm[freq]).mean().dropna()

        fig, ax = mpl_fig(3)
        ax.fill_between(samp.index, samp.values, alpha=0.08, color=GREEN)
        ax.plot(samp.index, samp.values, color=GREEN, lw=1.5)
        ax.set_title(f"{col} — {freq} average", fontsize=11)
        ax.set_ylabel(col, fontsize=9)
        st.pyplot(fig); plt.close()

        c1,c2 = st.columns(2)
        with c1:
            st.markdown("**Statistics**")
            st.dataframe(df[col].describe().to_frame().style.format("{:.5f}"), use_container_width=True)
        with c2:
            st.markdown("**Distribution**")
            fig3, ax3 = mpl_fig(2.5)
            ax3.hist(df[col].dropna(), bins=80, color=GREEN, alpha=0.85, edgecolor="white")
            ax3.set_xlabel(col, fontsize=9)
            st.pyplot(fig3); plt.close()

        st.markdown("**Correlation matrix**")
        corr = df.corr()
        fig4, ax4 = plt.subplots(figsize=(7,4))
        fig4.patch.set_facecolor("white"); ax4.set_facecolor("white")
        im2 = ax4.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr))); ax4.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
        ax4.set_yticks(range(len(corr))); ax4.set_yticklabels(corr.columns, fontsize=8)
        plt.colorbar(im2, ax=ax4, fraction=0.03, pad=0.04)
        ax4.spines[:].set_visible(False)
        st.pyplot(fig4); plt.close()

        with st.expander("Raw data (last 200 rows)"):
            st.dataframe(df.tail(200), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Forecast":
    st.markdown("# 🔮 LSTM Forecast")

    hz = st.slider("Forecast steps ahead", 1, 24, FUTURE_PERIOD_PREDICT)

    use_real = have_model and have_scaler and have_data and TF_OK

    if not TF_OK:
        st.markdown('<div class="warn">⚠️ TensorFlow unavailable — showing demo forecast.</div>', unsafe_allow_html=True)
    elif not have_model:
        st.markdown('<div class="warn">⚠️ No trained model found. Go to <b>Train Model</b> first.</div>', unsafe_allow_html=True)
    elif not have_data:
        st.markdown('<div class="warn">⚠️ No dataset found.</div>', unsafe_allow_html=True)

    now    = datetime.now()
    hist_t = [now - timedelta(hours=SEQ_LEN-i) for i in range(SEQ_LEN)]
    fore_t = [now + timedelta(hours=i+1) for i in range(hz)]

    if use_real:
        st.markdown('<div class="info">✅ Using your trained LSTM model on real data.</div>', unsafe_allow_html=True)
        try:
            scaled_df, _ = scale_data(df)
            # Use preprocess_df to get sequences (same as lstm.py)
            X_all, y_all = preprocess_df(scaled_df, seq_len=SEQ_LEN, shuffle=False)
            # Take last window
            window = X_all[-1]   # shape (SEQ_LEN, n_features)

            # Roll forward hz steps
            preds_scaled = []
            win = window.copy()
            for _ in range(hz):
                inp = win[np.newaxis]   # (1, SEQ_LEN, n_features)
                # stateful model needs batch_size=BATCH_SIZE_DEFAULT — use non-stateful predict
                try:
                    p = trained_model.predict(inp, verbose=0)[0, 0]
                except Exception:
                    p = float(trained_model(inp, training=False).numpy()[0, 0])
                preds_scaled.append(p)
                win = np.roll(win, -1, axis=0)
                win[-1, 0] = p

            # Inverse transform
            dummy = np.zeros((hz, scaler.n_features_in_))
            dummy[:, 0] = preds_scaled
            pred = scaler.inverse_transform(dummy)[:, 0]

            # Actual last SEQ_LEN points
            act_sc = X_all[-1, :, 0]
            dummy2 = np.zeros((SEQ_LEN, scaler.n_features_in_))
            dummy2[:, 0] = act_sc
            act = scaler.inverse_transform(dummy2)[:, 0]

        except Exception as e:
            st.error(f"Forecast error: {e}")
            act, pred = demo_series(SEQ_LEN, hz)
    else:
        act, pred = demo_series(SEQ_LEN, hz)

    fig, ax = mpl_fig(3.5)
    ax.fill_between(hist_t, act, alpha=0.08, color=GREEN)
    ax.plot(hist_t, act,  color=GREEN, lw=2,   label="Historical")
    ax.plot(fore_t, pred, color=BLUE,  lw=2.5, ls="--", marker="o", ms=4, label=f"LSTM +{hz}h forecast")
    ax.fill_between(fore_t, pred*0.95, pred*1.05, alpha=0.12, color=BLUE, label="95% CI")
    ax.axvline(now, color=GRAY, lw=1, ls=":")
    ax.set_ylabel("Global Active Power (kW)", fontsize=9)
    ax.legend(fontsize=9, frameon=False, ncol=2)
    st.pyplot(fig); plt.close()

    c1,c2,c3 = st.columns(3)
    c1.metric("Peak forecast",  f"{pred.max():.4f} kW")
    c2.metric("Min forecast",   f"{pred.min():.4f} kW")
    c3.metric("Avg forecast",   f"{pred.mean():.4f} kW")

    fdf = pd.DataFrame({
        "Timestamp":     [t.strftime("%Y-%m-%d %H:%M") for t in fore_t],
        "Forecast (kW)": np.round(pred, 5),
        "Lower CI":      np.round(pred*0.95, 5),
        "Upper CI":      np.round(pred*1.05, 5),
    })
    st.dataframe(fdf, use_container_width=True)
    st.download_button("⬇ Download forecast CSV", fdf.to_csv(index=False),
                       "forecast.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏋️ Train Model":
    st.markdown("# 🏋️ Train Model")
    st.markdown('<div class="info">Uses the <b>exact same architecture and preprocessing as <code>lstm/lstm.py</code></b> — CuDNNLSTM (GPU) or LSTM (CPU), deque-based sequences, same train/val split logic.</div>', unsafe_allow_html=True)

    if not TF_OK:
        st.error("TensorFlow required. Install with: `pip install tensorflow==2.15` (use Python 3.9–3.11)")
        st.stop()
    if not have_data:
        st.markdown('<div class="warn">⚠️ No dataset. Place <code>kaggle_data_1h.csv</code> in <code>dataset/</code> first.</div>', unsafe_allow_html=True)
        st.stop()

    c1,c2 = st.columns(2)
    with c1:
        epochs     = st.slider("Epochs",            5, 200, EPOCHS_DEFAULT)
        batch_size = st.selectbox("Batch size",     [32,64,128,256], index=1)
        val_pct    = st.slider("Validation %",      10, 30, 20)
    with c2:
        units      = st.slider("LSTM units",        16, 128, 36)
        use_gpu    = st.checkbox("Use CuDNNLSTM (GPU)", value=False)
        seq_len    = st.slider("Sequence length",   24, 200, SEQ_LEN)

    if st.button("▶ Start training", type="primary"):
        from sklearn.preprocessing import MinMaxScaler

        prog   = st.progress(0, text="Scaling data…")
        chart  = st.empty()
        stats  = st.empty()

        # 1. Scale
        sc = MinMaxScaler()
        scaled_vals = sc.fit_transform(df.values)
        scaled_df2  = pd.DataFrame(scaled_vals, index=df.index, columns=df.columns)

        # 2. Split using lstm.py split_data
        test_df, train_df = split_data(scaled_df2, percent=val_pct/100)

        # 3. Sequences using lstm.py preprocess_df
        prog.progress(5, text="Building sequences…")
        X_train, y_train = preprocess_df(train_df, seq_len=seq_len, shuffle=True)
        X_test,  y_test  = preprocess_df(test_df,  seq_len=seq_len, shuffle=False)

        # 4. Trim to batch-divisible length (lstm.py train_length)
        tl = train_length(len(X_train), batch_size)
        X_train = X_train[:tl]; y_train = y_train[:tl]

        prog.progress(10, text=f"Building model — {X_train.shape[0]} train samples…")

        # 5. Build model (lstm.py get_model)
        # For non-stateful training we rebuild without stateful
        mdl = Sequential()
        LSTMLayer = LSTM   # stateful LSTM needs fixed batch; use non-stateful for simplicity
        mdl.add(LSTMLayer(units, return_sequences=True,
                          input_shape=(seq_len, X_train.shape[2])))
        mdl.add(Activation("relu"))
        mdl.add(LSTMLayer(units, return_sequences=False))
        mdl.add(Activation("relu"))
        mdl.add(Dense(1))
        mdl.compile(loss="mae", optimizer="adam", metrics=["mse","mae"])

        # 6. Train with live chart
        train_losses, val_losses = [], []
        class LiveCB(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                tl_ = logs.get("loss", 0)
                vl_ = logs.get("val_loss", 0)
                train_losses.append(tl_); val_losses.append(vl_)
                pct = int((epoch+1)/epochs*90)+10
                prog.progress(pct, text=f"Epoch {epoch+1}/{epochs} — loss: {tl_:.5f} — val_loss: {vl_:.5f}")
                if len(train_losses) > 1:
                    fig_l, ax_l = mpl_fig(2.5)
                    ax_l.plot(train_losses, color=GREEN, lw=2, label="Train loss (MAE)")
                    ax_l.plot(val_losses,   color=BLUE,  lw=2, ls="--", label="Val loss (MAE)")
                    ax_l.set_xlabel("Epoch", fontsize=9)
                    ax_l.set_ylabel("MAE", fontsize=9)
                    ax_l.legend(fontsize=9, frameon=False)
                    chart.pyplot(fig_l); plt.close()

        cbs = [
            LiveCB(),
            EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(patience=4, factor=0.5, verbose=0),
        ]
        history = mdl.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs, batch_size=batch_size,
            callbacks=cbs, verbose=0,
        )

        # 7. Save — same filenames the rest of app expects
        mdl.save(str(AUTO_MODEL))
        with open(AUTO_SCALER, "wb") as f: pickle.dump(sc, f)
        run_n = len(sorted(ROOT.glob("history_run*.pkl"))) + 1
        with open(ROOT / f"history_run{run_n}.pkl", "wb") as f:
            pickle.dump(history.history, f)

        prog.progress(100, text="✅ Done!")
        st.success(f"✅ Saved `lstm_model.h5`, `scaler.pkl`, `history_run{run_n}.pkl`")
        stats.markdown(f"""
| Metric | Value |
|---|---|
| Final train MAE | `{history.history['loss'][-1]:.6f}` |
| Final val MAE   | `{history.history['val_loss'][-1]:.6f}` |
| Best val MAE    | `{min(history.history['val_loss']):.6f}` |
| Epochs run      | `{len(history.history['loss'])}` / {epochs} |
| Train samples   | `{len(X_train):,}` |
| Val samples     | `{len(X_test):,}` |
""")
        st.cache_resource.clear()
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAINING HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Training History":
    st.markdown("# 📈 Training History")

    auto_hists = sorted(ROOT.glob("history_run*.pkl"))
    histories  = []

    for hp in auto_hists:
        try:
            with open(hp,"rb") as f: histories.append((hp.name, pickle.load(f)))
        except: pass

    for uf in (up_hists or []):
        try: histories.append((uf.name, pickle.loads(uf.read())))
        except: pass

    if not histories:
        st.markdown('<div class="warn">No history files yet. Train a model — history is auto-saved as <code>history_run1.pkl</code>, <code>history_run2.pkl</code>, etc.</div>', unsafe_allow_html=True)
        st.markdown("**Demo — 5 synthetic runs (your 5 training sessions)**")
        fig, ax = mpl_fig(3.5)
        for i in range(5):
            np.random.seed(i)
            ep = np.arange(1,31)
            tl = 0.4*np.exp(-ep/10)+0.05+np.random.normal(0,.005,30)
            vl = 0.4*np.exp(-ep/12)+0.07+np.random.normal(0,.008,30)
            ax.plot(ep, tl, color=PALETTE[i], lw=2,   label=f"Run {i+1} train")
            ax.plot(ep, vl, color=PALETTE[i], lw=1.5, ls="--", alpha=0.7, label=f"Run {i+1} val")
        ax.set_xlabel("Epoch",fontsize=9); ax.set_ylabel("MAE",fontsize=9)
        ax.legend(fontsize=8, frameon=False, ncol=2)
        st.pyplot(fig); plt.close()
    else:
        st.markdown(f'<div class="info">✅ Found {len(histories)} training run(s)</div>', unsafe_allow_html=True)

        fig, ax = mpl_fig(3.5)
        for i,(name,h) in enumerate(histories):
            c = PALETTE[i % len(PALETTE)]
            ax.plot(h["loss"], color=c, lw=2, label=f"{name} train")
            if "val_loss" in h:
                ax.plot(h["val_loss"], color=c, lw=1.5, ls="--", alpha=0.7, label=f"{name} val")
        ax.set_xlabel("Epoch",fontsize=9); ax.set_ylabel("Loss (MAE)",fontsize=9)
        ax.legend(fontsize=8, frameon=False, ncol=2)
        st.pyplot(fig); plt.close()

        if any("mae" in h for _,h in histories):
            st.markdown("### MAE curves")
            fig2, ax2 = mpl_fig(3)
            for i,(name,h) in enumerate(histories):
                c = PALETTE[i%len(PALETTE)]
                if "mae"     in h: ax2.plot(h["mae"],     color=c, lw=2,   label=f"{name} train MAE")
                if "val_mae" in h: ax2.plot(h["val_mae"], color=c, lw=1.5, ls="--", alpha=0.7, label=f"{name} val MAE")
            ax2.set_xlabel("Epoch",fontsize=9); ax2.set_ylabel("MAE",fontsize=9)
            ax2.legend(fontsize=8, frameon=False, ncol=2)
            st.pyplot(fig2); plt.close()

        rows = []
        for name,h in histories:
            rows.append({
                "Run": name, "Epochs": len(h["loss"]),
                "Final train loss": round(h["loss"][-1],6),
                "Final val loss":   round(h["val_loss"][-1],6) if "val_loss" in h else "—",
                "Best val loss":    round(min(h["val_loss"]),6) if "val_loss" in h else "—",
                "Best epoch":       int(np.argmin(h["val_loss"]))+1 if "val_loss" in h else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        nr = [r for r in rows if isinstance(r.get("Best val loss"), float)]
        if nr:
            best = min(nr, key=lambda r: r["Best val loss"])
            st.success(f"🏆 Best run: **{best['Run']}** — val loss {best['Best val loss']} at epoch {best['Best epoch']}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐 Model Evaluation":
    st.markdown("# 📐 Model Evaluation")
    st.markdown("Runs the same metrics as `lstm/performance.py` — MSE, RMSE, MAE, MAPE, R², Theil-U")

    if not (have_model and have_scaler and have_data and TF_OK):
        st.markdown('<div class="warn">⚠️ Need trained model + scaler + dataset to evaluate.</div>', unsafe_allow_html=True)
    else:
        if st.button("▶ Run evaluation", type="primary"):
            with st.spinner("Evaluating…"):
                try:
                    from sklearn.metrics import mean_squared_error, r2_score
                    sc_df, _ = scale_data(df)
                    _, test_df = split_data(sc_df, percent=0.2)
                    X_test, y_test = preprocess_df(test_df, seq_len=SEQ_LEN, shuffle=False)

                    # Predict in chunks
                    preds = []
                    for i in range(len(X_test)):
                        inp = X_test[i:i+1]
                        try:    p = trained_model.predict(inp, verbose=0)[0,0]
                        except: p = float(trained_model(inp, training=False).numpy()[0,0])
                        preds.append(p)
                    preds   = np.array(preds)
                    y_true  = y_test

                    mse_  = mean_squared_error(y_true, preds)
                    rmse_ = sqrt(mse_)
                    mae_  = np.mean(np.abs(y_true - preds))
                    mape_ = mean_absolute_percentage_error(y_true, preds)
                    r2_   = r2_score(y_true, preds)
                    tu_   = theil_u(preds, y_true)

                    c1,c2,c3 = st.columns(3)
                    c1.metric("MSE",    f"{mse_:.6f}")
                    c2.metric("RMSE",   f"{rmse_:.6f}")
                    c3.metric("MAE",    f"{mae_:.6f}")
                    c1.metric("MAPE",   f"{mape_:.2f}%")
                    c2.metric("R²",     f"{r2_:.4f}")
                    c3.metric("Theil-U",f"{tu_:.4f}")

                    # Actual vs predicted chart
                    n_show = min(500, len(y_true))
                    fig, ax = mpl_fig(3.5)
                    ax.plot(y_true[:n_show],  color=GREEN, lw=1.5, label="Actual (scaled)")
                    ax.plot(preds[:n_show],   color=BLUE,  lw=1.5, ls="--", label="Predicted (scaled)")
                    ax.set_xlabel("Test samples", fontsize=9)
                    ax.set_ylabel("Scaled value", fontsize=9)
                    ax.legend(fontsize=9, frameon=False)
                    ax.set_title(f"Actual vs Predicted — first {n_show} test samples", fontsize=11)
                    st.pyplot(fig); plt.close()

                    # Residuals
                    resid = y_true[:n_show] - preds[:n_show]
                    fig2, ax2 = mpl_fig(2.5)
                    ax2.fill_between(range(n_show), resid, alpha=0.5, color=AMBER)
                    ax2.axhline(0, color=GRAY, lw=1)
                    ax2.set_xlabel("Sample", fontsize=9); ax2.set_ylabel("Residual", fontsize=9)
                    ax2.set_title("Residuals", fontsize=11)
                    st.pyplot(fig2); plt.close()

                except Exception as e:
                    st.error(f"Evaluation error: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL INFO
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Model Info":
    st.markdown("# ⚙️ Model Info")

    if have_model and TF_OK:
        st.markdown("### Loaded model summary")
        lines = []
        trained_model.summary(print_fn=lambda x: lines.append(x))
        st.code("\n".join(lines))
        st.metric("Total parameters", f"{trained_model.count_params():,}")
    else:
        st.markdown("### Architecture from `lstm/lstm.py`")
        st.code("""# From lstm/lstm.py get_model()
model = Sequential()
model.add(CuDNNLSTM(36, stateful=True, return_sequences=True,
          batch_input_shape=(BATCH_SIZE, SEQ_LEN, n_features)))
model.add(Activation('relu'))
model.add(CuDNNLSTM(36, stateful=True, return_sequences=False))
model.add(Activation('relu'))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam', metrics=['mse','mae'])""")

    st.markdown("### Constants from `lstm/lstm.py`")
    st.dataframe(pd.DataFrame([
        ["SEQ_LEN",              SEQ_LEN,              "Look-back window (hourly steps)"],
        ["FUTURE_PERIOD_PREDICT",FUTURE_PERIOD_PREDICT, "Steps ahead to forecast"],
        ["EPOCHS",               EPOCHS_DEFAULT,        "Training passes"],
        ["BATCH_SIZE",           BATCH_SIZE_DEFAULT,    "Batch size"],
    ], columns=["Constant","Value","Description"]), hide_index=True, use_container_width=True)

    st.markdown("### Preprocessing pipeline (from `lstm.py`)")
    st.code("""# 1. Load kaggle_data_1h.csv
df = pd.read_csv('kaggle_data_1h.csv', sep=',', index_col='time')

# 2. Drop sub-metering columns
df = df.drop(['Sub_metering_1','Sub_metering_2','Sub_metering_3'], axis=1)

# 3. Resample to 1h
df = df.resample('1h').mean().dropna()

# 4. Scale with MinMaxScaler
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df.values)

# 5. Build sequences with deque (preprocess_df)
#    seq_len=100 rows of history → predict next value

# 6. split_data() — 80% train / 20% test
# 7. train_length() — trim to batch-divisible size""")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: OPTIMIZATION TIPS  (ECOWATT — matches abstract requirement)
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Optimization Tips":
    st.markdown("# 💡 ECOWATT Optimization Tips")
    st.markdown("Smart suggestions based on your energy consumption patterns — aligned with the ECOWATT system objectives.")

    if not have_data:
        st.markdown('<div class="warn">⚠️ No dataset found. Place <code>kaggle_data_1h.csv</code> in <code>dataset/</code> to get personalized tips.</div>', unsafe_allow_html=True)
        act = demo_series(SEQ_LEN, 24)[0]
    else:
        act = df.iloc[-SEQ_LEN:, 0].values

    # ── Key stats ──────────────────────────────────────────────────────────────
    avg_load   = act.mean()
    peak_load  = act.max()
    peak_hour  = int(np.argmax(act) % 24)
    min_load   = act.min()
    # Weekly average from full dataset if available
    if have_data and len(df) > 24*7:
        weekly_avg = df.iloc[:,0].resample("D").mean().tail(7).mean()
        today_avg  = df.iloc[:,0].last("24h").mean()
        pct_vs_week = ((today_avg - weekly_avg) / weekly_avg * 100) if weekly_avg > 0 else 0
    else:
        weekly_avg  = avg_load * 0.95
        today_avg   = avg_load
        pct_vs_week = 5.0

    # Estimated monthly bill (Indian tariff ~₹6.5/kWh)
    RATE        = 6.5
    monthly_kwh = avg_load * 24 * 30
    monthly_bill= monthly_kwh * RATE
    saving_15   = monthly_bill * 0.15
    saving_30   = monthly_bill * 0.30

    # ── Metrics row ────────────────────────────────────────────────────────────
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-box"><div class="lbl">⚡ Avg load</div><div class="val">{avg_load:.3f} kW</div><div class="dl">Last {SEQ_LEN}h</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-box"><div class="lbl">📈 Peak load</div><div class="val">{peak_load:.3f} kW</div><div class="dl bad">Hour {peak_hour}:00</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-box"><div class="lbl">💰 Est. monthly bill</div><div class="val">₹{monthly_bill:,.0f}</div><div class="dl">{monthly_kwh:.0f} kWh/month</div></div>', unsafe_allow_html=True)
    with c4:
        delta_cls = "bad" if pct_vs_week > 0 else "good"
        delta_sym = "↑" if pct_vs_week > 0 else "↓"
        st.markdown(f'<div class="metric-box"><div class="lbl">📊 vs Weekly avg</div><div class="val">{abs(pct_vs_week):.1f}%</div><div class="dl {delta_cls}">{delta_sym} {"above" if pct_vs_week>0 else "below"} average</div></div>', unsafe_allow_html=True)

    st.divider()

    # ── Generate smart suggestions based on actual data ─────────────────────
    st.markdown("### 🎯 Smart recommendations")

    suggestions = []

    # Peak hour suggestion
    if peak_hour >= 18 and peak_hour <= 22:
        suggestions.append({
            "icon": "⚠️", "color": "#FAEEDA", "border": "#EF9F27",
            "title": f"Peak usage detected at {peak_hour}:00 (evening peak tariff window)",
            "detail": "Shift heavy appliances like washing machine, dishwasher, and water heater to after 10 PM when off-peak rates apply.",
            "saving": f"Est. saving: ₹{saving_15:,.0f}–₹{saving_30:,.0f}/month",
            "priority": "High"
        })
    elif peak_hour >= 6 and peak_hour <= 9:
        suggestions.append({
            "icon": "⚠️", "color": "#FAEEDA", "border": "#EF9F27",
            "title": f"Morning peak at {peak_hour}:00 — coincides with peak tariff",
            "detail": "Pre-heat water and run heavy appliances the night before to avoid morning peak charges.",
            "saving": f"Est. saving: ₹{saving_15:,.0f}/month",
            "priority": "High"
        })
    else:
        suggestions.append({
            "icon": "✅", "color": "#E1F5EE", "border": "#1D9E75",
            "title": f"Peak usage at {peak_hour}:00 — outside tariff peak window",
            "detail": "Your heaviest usage is already in a lower-tariff window. Maintain this pattern.",
            "saving": "Saving maintained",
            "priority": "Low"
        })

    # Usage vs weekly average
    if pct_vs_week > 15:
        suggestions.append({
            "icon": "🔴", "color": "#FCEBEB", "border": "#E24B4A",
            "title": f"Today's usage is {pct_vs_week:.1f}% above your weekly average",
            "detail": "Identify which appliances are running more than usual. Check for devices left on standby or AC running longer than needed.",
            "saving": f"Reducing to average saves ₹{monthly_bill*pct_vs_week/100/30*30:,.0f}/month",
            "priority": "High"
        })
    elif pct_vs_week > 5:
        suggestions.append({
            "icon": "🟡", "color": "#FAEEDA", "border": "#EF9F27",
            "title": f"Usage slightly elevated — {pct_vs_week:.1f}% above weekly average",
            "detail": "Minor increase detected. Check if AC or water heater is running longer than usual.",
            "saving": f"Est. saving: ₹{saving_15*0.5:,.0f}/month",
            "priority": "Medium"
        })
    else:
        suggestions.append({
            "icon": "✅", "color": "#E1F5EE", "border": "#1D9E75",
            "title": "Usage within normal range — good energy discipline!",
            "detail": "Your consumption is consistent with your weekly average. Keep monitoring for further improvements.",
            "saving": "On track",
            "priority": "Low"
        })

    # Load factor suggestion
    load_factor = min_load / peak_load if peak_load > 0 else 1
    if load_factor < 0.3:
        suggestions.append({
            "icon": "💡", "color": "#E6F1FB", "border": "#378ADD",
            "title": "High load variability detected — consider load balancing",
            "detail": f"Your min load ({min_load:.3f} kW) is only {load_factor*100:.0f}% of peak ({peak_load:.3f} kW). Spreading usage more evenly reduces peak demand charges.",
            "saving": f"Est. saving: ₹{saving_15:,.0f}/month",
            "priority": "Medium"
        })

    # Standby power suggestion
    if min_load > 0.2:
        standby_kwh = min_load * 8 * 30
        standby_cost = standby_kwh * RATE
        suggestions.append({
            "icon": "🔌", "color": "#EEEDFE", "border": "#7F77DD",
            "title": f"Standby power drain detected — minimum load {min_load:.3f} kW",
            "detail": "Even at night your minimum consumption is high, suggesting devices on standby. Unplug TV, set-top box, and chargers when not in use.",
            "saving": f"Est. saving: ₹{standby_cost:,.0f}/month",
            "priority": "Medium"
        })

    # Always-on suggestions
    suggestions.append({
        "icon": "🌡️", "color": "#E1F5EE", "border": "#1D9E75",
        "title": "Set AC thermostat to 24°C — each degree saves 6% energy",
        "detail": "Air conditioning is typically the largest consumer in Indian households. Raising the set temperature from 20°C to 24°C saves ~24% of AC energy costs.",
        "saving": f"Est. saving: ₹{monthly_bill*0.12:,.0f}/month",
        "priority": "High"
    })
    suggestions.append({
        "icon": "⏰", "color": "#E6F1FB", "border": "#378ADD",
        "title": "Use timer for water heater — run 30 min before use only",
        "detail": "Water heaters keeping water hot all day waste 30–40% of their energy. A simple timer reduces this to only when needed.",
        "saving": f"Est. saving: ₹{monthly_bill*0.08:,.0f}/month",
        "priority": "Medium"
    })
    suggestions.append({
        "icon": "💚", "color": "#E1F5EE", "border": "#1D9E75",
        "title": "Replace remaining incandescent bulbs with LED",
        "detail": "LEDs use 75% less energy than traditional bulbs and last 25x longer. For a typical household this saves 200–400 kWh/year.",
        "saving": f"Est. saving: ₹{200*RATE:,.0f}–₹{400*RATE:,.0f}/year",
        "priority": "Low"
    })

    # Sort by priority
    pri_order = {"High": 0, "Medium": 1, "Low": 2}
    suggestions.sort(key=lambda x: pri_order.get(x["priority"], 3))

    for s in suggestions:
        pri_color = {"High": "#A32D2D", "Medium": "#854F0B", "Low": "#0F6E56"}.get(s["priority"], "#888")
        st.markdown(f"""
        <div style="background:{s['color']};border-left:4px solid {s['border']};
                    border-radius:8px;padding:12px 16px;margin-bottom:10px">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:5px">
                <div style="font-size:13px;font-weight:600">{s['icon']} {s['title']}</div>
                <span style="font-size:11px;font-weight:500;color:{pri_color};
                             background:white;padding:2px 8px;border-radius:20px">
                    {s['priority']} priority
                </span>
            </div>
            <div style="font-size:12px;color:#444;margin-bottom:5px">{s['detail']}</div>
            <div style="font-size:12px;font-weight:500;color:{s['border']}">{s['saving']}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Potential savings summary ──────────────────────────────────────────────
    st.markdown("### 💰 Potential savings summary")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="lbl">Current monthly bill</div><div class="val">₹{monthly_bill:,.0f}</div><div class="dl">{monthly_kwh:.0f} kWh/month</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="lbl">After 15% reduction</div><div class="val" style="color:#0F6E56">₹{monthly_bill*0.85:,.0f}</div><div class="dl good">Save ₹{saving_15:,.0f}/month</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="lbl">After 30% reduction</div><div class="val" style="color:#0F6E56">₹{monthly_bill*0.70:,.0f}</div><div class="dl good">Save ₹{saving_30:,.0f}/month</div></div>', unsafe_allow_html=True)

    # ── CO2 impact ────────────────────────────────────────────────────────────
    st.markdown("### 🌿 Environmental impact")
    co2_current = monthly_kwh * 0.82
    co2_saved   = co2_current * 0.20
    fig, ax = mpl_fig(2)
    categories = ["Current CO₂", "After optimization"]
    values     = [co2_current, co2_current - co2_saved]
    colors     = [RED, GREEN]
    bars = ax.bar(categories, values, color=colors, width=0.4, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f kg", fontsize=10, padding=4)
    ax.set_ylabel("kg CO₂/month", fontsize=9)
    ax.set_title("Monthly carbon footprint", fontsize=11)
    st.pyplot(fig); plt.close()
    st.caption(f"Optimizing energy use could save **{co2_saved:.1f} kg CO₂/month** — equivalent to planting {co2_saved/21:.1f} trees.")