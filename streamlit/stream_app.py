import streamlit as st
import requests
import pandas as pd
import wfdb
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import tempfile
from pathlib import Path

API_URL = "http://localhost:8000/predict_from_upload"
WINDOW_SIZE = 216

st.set_page_config(layout="wide", page_title="ECG Analytics Dashboard")

def render_styled_metrics(total_windows, df_anomalies):
    """
    Render custom KPI cards using HTML/CSS instead of default Streamlit metrics.
    """
    count_F = len(df_anomalies[df_anomalies["predicted_class"] == 0])
    count_Q = len(df_anomalies[df_anomalies["predicted_class"] == 1])
    count_S = len(df_anomalies[df_anomalies["predicted_class"] == 2])
    count_V = len(df_anomalies[df_anomalies["predicted_class"] == 3])

    html_content = f"""
    <div style="display: flex; gap: 15px; margin-bottom: 20px;">
        <div style="flex: 1; padding: 15px; border-radius: 6px; background-color: #262730; color: #FAFAFA; border-left: 6px solid #888;">
            <div style="font-size: 0.9rem; color: #ccc;">All windows</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{total_windows}</div>
        </div>
        <div style="flex: 1; padding: 15px; border-radius: 6px; background-color: #262730; color: #FAFAFA; border-left: 6px solid #FF4B4B;">
            <div style="font-size: 0.9rem; color: #ccc;">Class F (Fusion)</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{count_F}</div>
        </div>
        <div style="flex: 1; padding: 15px; border-radius: 6px; background-color: #262730; color: #FAFAFA; border-left: 6px solid #FFA500;">
            <div style="font-size: 0.9rem; color: #ccc;">Class Q (Unknown)</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{count_Q}</div>
        </div>
        <div style="flex: 1; padding: 15px; border-radius: 6px; background-color: #262730; color: #FAFAFA; border-left: 6px solid #1E90FF;">
            <div style="font-size: 0.9rem; color: #ccc;">Class S (Supraventricular)</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{count_S}</div>
        </div>
        <div style="flex: 1; padding: 15px; border-radius: 6px; background-color: #262730; color: #FAFAFA; border-left: 6px solid #8A2BE2;">
            <div style="font-size: 0.9rem; color: #ccc;">Class V (Ventricular)</div>
            <div style="font-size: 1.8rem; font-weight: bold;">{count_V}</div>
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)

@st.fragment
def render_morphology_viewer(signal, anomalies_df):
    """
    Overlay beats from the selected class to visualize morphology variance.
    """
    st.markdown("**Morphology analysis (Beat Template Overlay)**")
    
    if anomalies_df.empty:
        st.info("No anomalies available for morphology analysis.")
        return

    available_classes = anomalies_df["Visual_Label"].unique()
    selected_label = st.selectbox("Select class", available_classes, label_visibility="collapsed")
    
    subset = anomalies_df[anomalies_df["Visual_Label"] == selected_label]
    
    fig = go.Figure()
    all_beats = []
    
    for _, row in subset.iterrows():
        start = int(row["start_sample"])
        w_size = int(row["window_size"])
        
        if start >= 0 and start + w_size < len(signal):
            beat = signal[start : start + w_size]
            all_beats.append(beat)
            
            fig.add_trace(go.Scatter(
                y=beat, mode="lines",
                line=dict(color="gray", width=1),
                opacity=0.15,
                hoverinfo="skip"
            ))
            
    if all_beats:
        mean_beat = np.mean(all_beats, axis=0)
        fig.add_trace(go.Scatter(
            y=mean_beat, mode="lines",
            name="Mean morphology",
            line=dict(color="black", width=3)
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        plot_bgcolor='white',
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=True, gridcolor='#eee', zeroline=False, title="Amplitude")
    )
    
    st.plotly_chart(fig, use_container_width=True)


@st.fragment
def render_interactive_viewer(signal, fs, predictions, total_duration):
    col_title, col_width = st.columns([3, 1])
    with col_title:
        st.subheader("Signal visualization with predictions")
    with col_width:
        duration_sec = st.selectbox("View width [s]", [5, 10, 30, 60], index=1)

    chart_placeholder = st.empty()

    start_sec = st.slider(
        "View start [s]",
        min_value=0.0,
        max_value=float(total_duration),
        step=1.0,
        label_visibility="collapsed",
        key="chart_start_slider"
    )

    fig = plot_signal_segment(
        signal=signal,
        fs=fs,
        predictions=predictions,
        start_sec=start_sec,
        duration_sec=duration_sec,
        window_size=WINDOW_SIZE,
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)


def load_signal_from_upload(hea_bytes: bytes, dat_bytes: bytes, base_name: str, channel: int):
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            (tmp_path / f"{base_name}.hea").write_bytes(hea_bytes)
            (tmp_path / f"{base_name}.dat").write_bytes(dat_bytes)

            record = wfdb.rdrecord(str(tmp_path / base_name))
            signal = record.p_signal[:, channel]
            fs = record.fs
            return signal, fs
    except Exception as e:
        st.error(f"Error loading uploaded WFDB files: {e}")
        return None, None

def fetch_predictions_from_upload(hea_file, dat_file, channel: int):
    files = {
        "hea_file": (hea_file.name, hea_file.getvalue(), "application/octet-stream"),
        "dat_file": (dat_file.name, dat_file.getvalue(), "application/octet-stream"),
    }
    data = {"channel": str(channel)}
    response = requests.post(API_URL, files=files, data=data, timeout=30)

    if response.status_code == 503:
        st.error("Service unavailable: the model or scaler has not been loaded into container memory.")
        return None
    if response.status_code == 400:
        st.error(f"Uploaded file error on the backend: {response.json().get('detail')}")
        return None

    response.raise_for_status()
    return response.json()

def normalize_api_predictions(preds):
    if not isinstance(preds, dict):
        return preds
    results = preds.get("results", preds)
    if not isinstance(results, list):
        return preds

    normalized = []
    for item in results:
        if not isinstance(item, dict):
            continue
        peak_sample = item.get("peak_sample")
        normalized.append(
            {
                **item,
                "window_index": item.get("beat_index"),
                "start_sample": None if peak_sample is None else int(peak_sample) - (WINDOW_SIZE // 2),
                "window_size": WINDOW_SIZE,
            }
        )
    return normalized


def plot_signal_segment(signal, fs, predictions, start_sec, duration_sec, window_size):
    start_idx = int(start_sec * fs)
    end_idx = int((start_sec + duration_sec) * fs)
    end_idx = min(end_idx, len(signal))

    y_data = signal[start_idx:end_idx]
    x_data = np.arange(start_idx, end_idx) / fs

    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=y_data,
            mode="lines",
            name="ECG signal",
            line=dict(color="black", width=1.5),
        )
    )

    color_map = {0: "#FF4B4B", 1: "#FFA500", 2: "#1E90FF", 3: "#8A2BE2"}
    label_map = {0: "F", 1: "Q", 2: "S", 3: "V"}

    for pred in predictions:
        w_start = pred.get("start_sample")
        if w_start is None:
            continue

        w_end = w_start + window_size

        if w_end > start_idx and w_start < end_idx:
            if not pred.get("is_anomaly", False):
                continue
            p_class = pred.get("predicted_class")
            if p_class is None or str(p_class).strip().lower() == "none":
                continue

            color = color_map.get(p_class, "gray")
            label_text = label_map.get(p_class, f"{p_class}")

            x0_sec = max(w_start, start_idx) / fs
            x1_sec = min(w_end, end_idx) / fs

            fig.add_shape(
                type="rect",
                xref="x", yref="paper",
                x0=x0_sec, x1=x1_sec,
                y0=0.0, y1=0.05,
                fillcolor=color,
                line_width=0,
                layer="above"
            )
            
            fig.add_annotation(
                x=x0_sec, y=0.07, xref="x", yref="paper",
                text=label_text, showarrow=False,
                font=dict(color=color, size=11, family="Arial Black"),
                xanchor="left"
            )

    fig.update_layout(
        height=500,
        plot_bgcolor='white',
        xaxis=dict(
            title="Time [s]",
            tickmode='linear',
            dtick=0.2,
            minor=dict(dtick=0.04, gridcolor='rgba(255, 100, 100, 0.3)', gridwidth=1),
            gridcolor='rgba(255, 100, 100, 0.7)',
            gridwidth=1.5,
            zeroline=False,
            showgrid=True
        ),
        yaxis=dict(
            title="Amplitude [mV]",
            tickmode='linear',
            dtick=0.5,
            minor=dict(dtick=0.1, gridcolor='rgba(255, 100, 100, 0.3)', gridwidth=1),
            gridcolor='rgba(255, 100, 100, 0.7)',
            gridwidth=1.5,
            zeroline=False,
            showgrid=True
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    return fig


st.title("Clinical Interface: Inception-Conformer ECG")

with st.sidebar:
    st.header("Input configuration")
    channel = st.number_input("Channel", min_value=0, max_value=5, value=0, step=1)
    uploaded_files = st.file_uploader(
        "Upload WFDB files (.hea and .dat)",
        type=["hea", "dat"],
        accept_multiple_files=True,
    )
    run_btn = st.button("Run API prediction", type="primary")

if "predictions" not in st.session_state:
    st.session_state.predictions = None
    st.session_state.current_signal = None
    st.session_state.current_fs = None
if "chart_start_slider" not in st.session_state:
    st.session_state.chart_start_slider = 0.0
if "last_selected_row" not in st.session_state:
    st.session_state.last_selected_row = None

if run_btn:
    with st.spinner("Processing FastAPI container..."):
        try:
            if not uploaded_files or len(uploaded_files) != 2:
                st.error("Upload exactly two files: one .hea and one .dat.")
                st.stop()

            file_map = {Path(file.name).suffix.lower(): file for file in uploaded_files}
            hea_file = file_map.get(".hea")
            dat_file = file_map.get(".dat")

            if hea_file is None or dat_file is None:
                st.error("Upload both a .hea file and a .dat file.")
                st.stop()

            preds = fetch_predictions_from_upload(hea_file, dat_file, channel)
            if not preds:
                st.stop()

            normalized_preds = normalize_api_predictions(preds)
            if not isinstance(normalized_preds, list):
                st.error(f"Critical data structure error. Expected a list, got: {type(normalized_preds).__name__}")
                st.stop()

            signal, fs = load_signal_from_upload(
                hea_file.getvalue(),
                dat_file.getvalue(),
                Path(hea_file.name).stem,
                channel,
            )

            if signal is not None:
                st.session_state.current_signal = signal
                st.session_state.current_fs = fs

            st.session_state.predictions = normalized_preds
            st.session_state.chart_start_slider = 0.0
            st.session_state.last_selected_row = None
            st.success("Inference completed.")
            
        except requests.exceptions.ConnectionError:
            st.error("Connection refused. Check whether the backend container is running on port 8000.")
        except Exception as e:
            st.error(f"Critical communication error: {e}")

if st.session_state.predictions and st.session_state.current_signal is not None and st.session_state.current_fs is not None:
    signal = st.session_state.current_signal
    fs = st.session_state.current_fs
    total_duration = len(signal) / fs

    df = pd.DataFrame(st.session_state.predictions)
    anomalies_df = df[df["is_anomaly"] == True].copy()
    
    display_df = pd.DataFrame()
    if not anomalies_df.empty:
        ui_label_map = {0: "🟥 F", 1: "🟧 Q", 2: "🟦 S", 3: "🟪 V"}
        anomalies_df["Visual_Label"] = anomalies_df["predicted_class"].map(ui_label_map)
        anomalies_df["time_sec"] = anomalies_df["start_sample"] / fs
        
        display_df = anomalies_df[
            ["window_index", "time_sec", "start_sample", "window_size", "binary_anomaly_probability", "Visual_Label"]
        ].reset_index(drop=True)

    if "anomaly_table" in st.session_state:
        selected_rows = st.session_state.anomaly_table.get("selection", {}).get("rows", [])
        current_selection = selected_rows[0] if selected_rows else None
        
        if current_selection is not None and current_selection != st.session_state.last_selected_row:
            selected_sec = display_df.iloc[current_selection]["time_sec"]
            st.session_state.chart_start_slider = max(0.0, float(selected_sec) - 1.0)
            st.session_state.last_selected_row = current_selection
        elif not selected_rows:
            st.session_state.last_selected_row = None

    render_interactive_viewer(
        signal=signal,
        fs=fs,
        predictions=st.session_state.predictions,
        total_duration=total_duration,
    )

    st.markdown("---")
    
    render_styled_metrics(len(df), anomalies_df)

    if not display_df.empty:
        col_pie, col_table = st.columns([1, 1.5])
        
        with col_pie:
            st.markdown("**Arrhythmia type distribution**")
            label_map = {0: "F", 1: "Q", 2: "S", 3: "V"}
            anomalies_df["Label"] = anomalies_df["predicted_class"].map(label_map)
            fig_pie = px.pie(
                anomalies_df,
                names="Label",
                hole=0.4,
                color_discrete_map={"F":"#FF4B4B", "Q":"#FFA500", "S":"#1E90FF", "V":"#8A2BE2"},
            )
            fig_pie.update_layout(margin=dict(t=10, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            render_morphology_viewer(signal, display_df)

        with col_table:
            st.markdown("**Event log**")
            table_view = display_df.drop(columns=["window_size"])
            
            st.dataframe(
                table_view,
                key="anomaly_table",
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "window_index": "Window ID",
                    "time_sec": st.column_config.NumberColumn("Time [s]", format="%.3f"),
                    "start_sample": "Sample index",
                    "binary_anomaly_probability": st.column_config.ProgressColumn(
                        "Detection confidence", format="%.2f", min_value=0, max_value=1
                    ),
                    "Visual_Label": "Classification",
                },
                hide_index=True,
                use_container_width=True,
                height=550,
            )
    else:
        st.success("No anomalies were found in the analyzed signal segment.")