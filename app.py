"""Streamlit Web Application"""

import pandas as pd
import streamlit as st

from plotly.graph_objs import Figure, Scatter

from src.ingest import query_ids
from src.pipeline import ForecastingPipeline


def generate_forecast(query_id: str) -> tuple[int, pd.DataFrame, pd.Series]:
    """Returns the forecast horizon, a pd.DataFrame containing the in-sample features and
    stationary target, and a pd.Series containing the original in-sample target and forecast

    Args:
        query_id (str): ID that's used to query a subset of raw data from Postgres

    Returns:
        tuple[int, pd.DataFrame, pd.Series]: Forecast horizon (int), in-sample features and
        stationary target (pd.DataFrame), original in-sample target and forecast (pd.Series)
    """
    try:
        fp: ForecastingPipeline = ForecastingPipeline(query_id)
        forecast: pd.Series = fp.run()
        return (
            forecast.shape[0],
            pd.concat((fp.features, fp.labels), axis=1),
            pd.concat((fp.stationary_data["energy_demand"], forecast), axis=0)
        )
    except Exception as e:
        raise e


st.title("Energy Demand Forecasting :clock9:")
query_id: str = st.selectbox("Select the dataset ID", query_ids())
horizon, train_data, series = generate_forecast(query_id)
series.name = "energy_demand_mw"

st.write(f"""#### {query_id.upper()} Hourly Energy Demand""")
fig: Figure = Figure()
fig.add_trace(
    Scatter(
        x=series.iloc[:-horizon].index,
        y=series.iloc[:-horizon],
        name="In-sample data",
        mode="lines",
        line={"color": "aqua", "width": 2},
    )
)
fig.add_trace(
    Scatter(
        x=series.iloc[-horizon:].index,
        y=series.iloc[-horizon:],
        name="Forecast",
        mode="lines",
        line={"color": "aqua", "width": 4, "dash": "dot"},
    )
)
fig.update_layout(
    autosize=True,
    width=1400,
    height=600,
    xaxis={
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "tickfont": {"family": "Arial", "size": 14},
    },
    xaxis_rangeslider_visible=True,
    yaxis={
        "showline": True,
        "showgrid": False,
        "showticklabels": True,
        "tickfont": {"family": "Arial", "size": 14},
    },
    showlegend=False,
)
fig.update_xaxes(
    title_text="Timestamp (UTC)", title_font={"size": 16, "family": "Arial"}, title_standoff=20
)
fig.update_yaxes(
    title_text="Energy Demand (MW)",
    title_font={"size": 16, "family": "Arial"},
    title_standoff=20,
)
st.plotly_chart(fig)

if st.checkbox("Original Time Series & Forecast"):
    st.dataframe(
        series.rename_axis("timestamp_utc")
        .to_frame()
        .style.format(precision=0)
        .applymap(
            lambda _: "background-color: teal", subset=(series.tail(horizon).index, slice(None))
        )
    )

if st.checkbox("Training Data"):
    st.dataframe(train_data.rename_axis("timestamp_utc"))
