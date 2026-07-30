import streamlit as st
import requests
import pandas as pd
import wfdb
import plotly.graph_objects as go
import numpy as np
import plotly.express as px

# Environment configuration
API_URL = "http://localhost:8000/predict_from_physionet"

st.set_page_config(layout="wide", page_title="ECG Analytics Dashboard")


@st.fragment
def render_interactive_viewer(signal, fs, predictions, total_duration, stride):
    """
    Isolated UI component. Changing the slider or selectbox only reruns this function.
    """
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
        value=0.0,
        step=1.0,
        label_visibility="collapsed",
    )

    fig = plot_signal_segment(
        signal=signal,
        fs=fs,
        predictions=predictions,
        start_sec=start_sec,
        duration_sec=duration_sec,
        window_size=stride,
    )

    chart_placeholder.plotly_chart(fig, use_container_width=True)


@st.cache_data
def load_signal(record_name: str, pn_dir: str, channel: int):
    """
    Loads a signal from the PhysioNet database.
    The data are cached in memory so the navigation slider does not
    force repeated downloads over the network.
    """
    try:
        record = wfdb.rdrecord(record_name, pn_dir=pn_dir)
        signal = record.p_signal[:, channel]
        fs = record.fs
        return signal, fs
    except Exception as e:
        st.error(f"Error loading data from WFDB: {e}")
        return None, None


def fetch_predictions(record_name: str, pn_dir: str, channel: int, stride: int):
    """
    Sends a POST request with a JSON payload to the FastAPI container.
    """
    payload = {
        "record_name": record_name,
        "pn_dir": pn_dir,
        "channel": channel,
        "stride": stride,
    }

    response = requests.post(API_URL, json=payload, timeout=30)

    if response.status_code == 503:
        st.error("Service unavailable: the model or scaler has not been loaded into container memory.")
        return None
    if response.status_code == 400:
        st.error(f"PhysioNet data retrieval error on the backend: {response.json().get('detail')}")
        return None

    response.raise_for_status()
    return response.json()


def plot_signal_segment(signal, fs, predictions, start_sec, duration_sec, window_size):
    """
    Renders an ECG signal segment with prediction windows overlaid.
    Uses the 'is_anomaly' key for filtering and maps class IDs to text labels.
    """
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
            line=dict(color="black", width=1),
        )
    )

    # Visual configuration
    color_map = {0: "red", 1: "orange", 2: "blue", 3: "purple"}

    # Class mapping according to the selected notation
    label_map = {0: "F", 1: "Q", 2: "S", 3: "V"}

    for pred in predictions:
        w_start = pred.get("start_sample")
        if w_start is None:
            continue

        w_end = w_start + window_size

        if w_end > start_idx and w_start < end_idx:
            # API contract key: 'is_anomaly'
            if not pred.get("is_anomaly", False):
                continue

            p_class = pred.get("predicted_class")
            if p_class is None or str(p_class).strip().lower() == "none":
                continue

            color = color_map.get(p_class, "gray")
            # Get the text label; if the class is missing from the mapping, fall back to the class ID
            label_text = label_map.get(p_class, f"Class {p_class}")

            x0_sec = max(w_start, start_idx) / fs
            x1_sec = min(w_end, end_idx) / fs

            fig.add_vrect(
                x0=x0_sec,
                x1=x1_sec,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=1,
                line_color=color,
                annotation_text=label_text,
                annotation_position="top left",
            )

    fig.update_layout(
        height=500,
        xaxis_title="Time [s]",
        yaxis_title="Amplitude [mV]",
        margin=dict(l=0, r=0, t=30, b=0),
        showlegend=False,
    )
    return fig


# --- Main application interface ---

st.title("Analytical Interface: Inception-Conformer ECG")

with st.sidebar:
    st.header("Input parameters")
    record_name = st.text_input("Record name", value="100")
    pn_dir = st.text_input("Directory", value="mitdb")
    channel = st.number_input("Channel", min_value=0, max_value=5, value=0, step=1)
    stride = st.number_input("Window step (stride)", min_value=10, max_value=2000, value=216, step=1)

    run_btn = st.button("Run API prediction", type="primary")

# Session state initialization
if "predictions" not in st.session_state:
    st.session_state.predictions = None
    st.session_state.current_record = None

# Run button handling
if run_btn:
    with st.spinner("Querying the FastAPI container..."):
        try:
            preds = fetch_predictions(record_name, pn_dir, channel, stride)
            if preds:
                # Unwrap any top-level key depending on the StreamResponse format
                # Analyze and normalize the response structure
                normalized_preds = preds

                if isinstance(preds, dict):
                    # If this is a single window (a dictionary with model keys), force a list
                    if "start_sample" in preds:
                        normalized_preds = [preds]
                    # Check the most common keys for result lists
                    elif "predictions" in preds:
                        normalized_preds = preds["predictions"]
                    elif "results" in preds:
                        normalized_preds = preds["results"]
                    elif "data" in preds:
                        normalized_preds = preds["data"]

                # Type guard before passing data to the chart
                if not isinstance(normalized_preds, list):
                    st.error(
                        f"Critical data structure error. Expected a list, got: {type(normalized_preds).__name__}"
                    )
                    st.write("Raw API object snapshot:", preds)
                    st.stop()  # Stops further UI rendering and prevents a TypeError

                st.session_state.predictions = normalized_preds
                st.session_state.current_record = (record_name, pn_dir, channel)
                st.success("Inference completed successfully.")
        except requests.exceptions.ConnectionError:
            st.error("Connection refused. Check whether the backend container is running on port 8000.")
        except Exception as e:
            st.error(f"Critical communication error: {e}")

# Display results if they are present in the session
if st.session_state.predictions and st.session_state.current_record:
    curr_record_name, curr_pn_dir, curr_channel = st.session_state.current_record

    signal, fs = load_signal(curr_record_name, curr_pn_dir, curr_channel)

    if signal is not None:
        total_duration = len(signal) / fs

        # Call the isolated fragment
        render_interactive_viewer(
            signal=signal,
            fs=fs,
            predictions=st.session_state.predictions,
            total_duration=total_duration,
            stride=stride,
        )

        # Data availability guard
if st.session_state.predictions:
    st.markdown("---")
    st.subheader("Statistics and detection analysis")

    # Convert to a DataFrame for fast vectorized aggregation
    df = pd.DataFrame(st.session_state.predictions)

    # Metrics panel (KPI)
    total_windows = len(df)
    # Extract only anomalies for detailed analysis
    anomalies_df = df[df["is_anomaly"] == True].copy()
    anomalies_count = len(anomalies_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("Analyzed windows", total_windows)
    col2.metric("Detected anomalies", anomalies_count)
    col3.metric(
        "Anomaly rate",
        f"{(anomalies_count / total_windows) * 100:.1f}%" if total_windows > 0 else "0%",
    )

    st.write("")  # Spacer

    # Detailed analysis panel (chart + table)
    col_pie, col_table = st.columns([1, 1.5])

    if not anomalies_df.empty:
        # Base mapping for logic and charts
        label_map = {0: "F", 1: "Q", 2: "S", 3: "V"}
        anomalies_df["Label"] = anomalies_df["predicted_class"].map(label_map)

        # UI mapping with Unicode glyphs (colored squares)
        # Red, orange, blue, purple
        ui_label_map = {0: "🟥 F", 1: "🟧 Q", 2: "🟦 S", 3: "🟪 V"}
        anomalies_df["Visual_Label"] = anomalies_df["predicted_class"].map(ui_label_map)

        # Left column: pie chart
        with col_pie:
            st.markdown("**Anomaly type distribution**")
            fig_pie = px.pie(
                anomalies_df,
                names="Label",
                hole=0.4,
                color_discrete_sequence=["red", "orange", "blue", "purple"],
            )
            fig_pie.update_layout(margin=dict(t=20, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)

        # Right column: formatted table
        with col_table:
            st.markdown("**Detected events register**")

            # Use the column with embedded Unicode squares
            display_df = anomalies_df[
                ["window_index", "start_sample", "binary_anomaly_probability", "Visual_Label"]
            ].copy()

            st.dataframe(
                display_df,
                column_config={
                    "window_index": "Window ID",
                    "start_sample": "Start sample",
                    "binary_anomaly_probability": st.column_config.ProgressColumn(
                        "Detection confidence",
                        format="%.2f",
                        min_value=0,
                        max_value=1,
                    ),
                    "Visual_Label": "Class",
                },
                hide_index=True,
                use_container_width=True,
                height=350,
            )
    else:
        st.success("No anomalies were found in the analyzed signal.")