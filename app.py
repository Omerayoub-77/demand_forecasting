import hashlib 
import io
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
# ============================================================================
# AUTHENTICATION
# ============================================================================
SHARED_PASSWORD = "powergrid2025"
PASSWORD_HASH = hashlib.sha256(SHARED_PASSWORD.encode()).hexdigest()
USERS_DB = {
        "omer": PASSWORD_HASH,
    "ahmed": PASSWORD_HASH,
    "raqeeb": PASSWORD_HASH,
}
USER_ROLES = {
        "omer": "Lead Analyst",
    "ahmed": "Demand Planner",
    "raqeeb": "Procurement Manager",
}
DEFAULT_STOCK = {
        "ACSR_70": 620,
    "Tower_Bolts": 420,
    "Insulators": 350,
    "Transformer_Oil": 280,
    "Copper_Wire": 300,
    "Pole_Hardware": 390,
}
DEFAULT_UNIT_COST = {
        "ACSR_70": 18.5,
    "Tower_Bolts": 4.2,
    "Insulators": 9.5,
    "Transformer_Oil": 16.0,
    "Copper_Wire": 14.8,
    "Pole_Hardware": 7.1,
}
def init_auth_db():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.login_time = None
        st.session_state.page = "Overview"
def login_user(username, password):
    if not username or not password:
        return False, "Username and password cannot be empty."
    if username not in USERS_DB:
        return False, f"User '{username}' not found in system."
    pwd_hash = hashlib.sha256(password.encode()).hexdigest()
    if pwd_hash != USERS_DB[username]:
        return False, "Incorrect password. Please try again."
    st.session_state.logged_in = True
    st.session_state.username = username
    st.session_state.user_role = USER_ROLES.get(username, "User")
    st.session_state.login_time = datetime.now()
    st.session_state.page = "Overview"
    return True, f"Welcome, {username}."
def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.login_time = None
    st.session_state.page = "Overview"
def render_login_screen():
    st.markdown(
            """
        <div class="hero-card">
            <div class="hero-eyebrow">Operations Intelligence</div>
            <div class="hero-title">PowerGrid Material Demand Forecasting</div>
            <p class="hero-copy">
                A focused workspace for demand forecasting, inventory risk visibility, and procurement planning.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left_col, right_col = st.columns([1.7, 1])
    with left_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Secure Team Access</div>', unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Access dashboard", use_container_width=True)
            if submitted:
                success, message = login_user(username, password)
                if success:
                    st.rerun()
                else:
                    st.error(message)
        st.markdown("</div>", unsafe_allow_html=True)
    with right_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Authorized Users</div>', unsafe_allow_html=True)
        team_df = pd.DataFrame(
                {"Username": list(USER_ROLES.keys()), "Role": list(USER_ROLES.values())}
        )
        st.dataframe(team_df, hide_index=True, use_container_width=True)
        st.markdown(
                "<p class='small-note'>Shared password for the demo environment: powergrid2025</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
# ============================================================================
# FORECAST DATA AND ENGINE
# ============================================================================
def create_synthetic_data(start_date="2025-08-01", end_date="2026-04-30", num_materials=6):
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    materials = [
            "ACSR_70",
        "Tower_Bolts",
        "Insulators",
        "Transformer_Oil",
        "Copper_Wire",
        "Pole_Hardware",
    ][:num_materials]
    data = []
    for material in materials:
        base_demand = np.random.uniform(15, 50)
        trend = np.linspace(0, 10, len(date_range))
        seasonality = 5 * np.sin(np.arange(len(date_range)) * 2 * np.pi / 365)
        noise = np.random.normal(0, 2, len(date_range))
        demand = np.maximum(base_demand + trend + seasonality + noise, 5)
        for date, value in zip(date_range, demand):
            data.append({"Date": date, "Material": material, "Demand": value})
    return pd.DataFrame(data)
def aggregate_to_weekly(df):
    weekly_source = df.copy()
    weekly_source["Week"] = weekly_source["Date"].dt.isocalendar().week
    weekly_source["Year"] = weekly_source["Date"].dt.isocalendar().year
    weekly_df = (
            weekly_source.groupby(["Year", "Week", "Material"])
        .agg(Demand=("Demand", "sum"), Days_In_Week=("Date", "nunique"))
        .reset_index()
        .sort_values(["Material", "Year", "Week"])
        .reset_index(drop=True)
    )
    weekly_df = weekly_df[weekly_df["Days_In_Week"] == 7].copy()
    return weekly_df.drop(columns=["Days_In_Week"]).reset_index(drop=True)
def simple_forecast_material(weekly_df, material, n_weeks=12):
    mat = weekly_df[weekly_df["Material"] == material].sort_values(["Year", "Week"]).reset_index(drop=True)
    if len(mat) < 3:
        now = datetime.now()
        dates = [now + timedelta(weeks=i + 1) for i in range(n_weeks)]
        return (
                pd.DataFrame(
                    {
                        "Date": dates,
                    "Year": [now.isocalendar().year] * n_weeks,
                    "Week": [(now.isocalendar().week + i) % 52 + 1 for i in range(n_weeks)],
                    "Forecast": [0.0] * n_weeks,
                    "Lower_Band": [0.0] * n_weeks,
                    "Upper_Band": [0.0] * n_weeks,
                }
            ),
            0.0,
            0.0,
        )
    y = mat["Demand"].values
    t = np.arange(len(y))
    coeffs = np.polyfit(t, y, 1)
    trend_full = np.polyval(coeffs, np.arange(len(y) + n_weeks))
    residuals = y - trend_full[: len(y)]
    week_nums = mat["Week"].astype(int).values
    seasonal = {
            week: float(np.mean(residuals[week_nums == week])) if np.any(week_nums == week) else 0.0
        for week in range(1, 53)
    }
    last_year = int(mat["Year"].iloc[-1])
    last_week = int(mat["Week"].iloc[-1])
    future = []
    future_year = last_year
    future_week = last_week
    for _ in range(n_weeks):
        future_week += 1
        if future_week > 52:
            future_week = 1
            future_year += 1
        future.append((future_year, future_week))
    future_trend = trend_full[len(y) :]
    future_season = np.array([seasonal[week] for _, week in future])
    forecast = np.maximum(future_trend + future_season, 0)
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mean_actual = float(np.mean(y)) if len(y) else 1.0
    mape = float(np.mean(np.abs(residuals) / np.maximum(np.abs(y), 1))) * 100
    band = 1.64 * rmse
    dates = [datetime.fromisocalendar(year, week, 1) for year, week in future]
    forecast_df = pd.DataFrame(
            {
                "Date": dates,
            "Year": [year for year, _ in future],
            "Week": [week for _, week in future],
            "Forecast": forecast,
            "Lower_Band": np.maximum(forecast - band, 0),
            "Upper_Band": forecast + band,
            "Bias_vs_Mean_%": ((forecast - mean_actual) / max(mean_actual, 1.0)) * 100,
        }
    )
    return forecast_df, rmse, mape
def create_pdf_report_bytes(df, weekly_df, username, role, forecast_summary):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        return None, "reportlab not installed"
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(50, 760, "PowerGrid Material Demand Forecast Report")
    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, 742, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    pdf.drawString(50, 728, f"Generated by: {username} ({role})")
    pdf.drawString(50, 714, f"Materials tracked: {len(weekly_df['Material'].unique())}")
    pdf.drawString(50, 700, f"Historical period: {df['Date'].min().date()} to {df['Date'].max().date()}")
    y_pos = 672
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(50, y_pos, "Forecast Snapshot")
    y_pos -= 18
    pdf.setFont("Helvetica", 10)
    for _, row in forecast_summary.iterrows():
        line = (
                f"{row['Material']}: {row['Forecast_4W']:.0f} units in 4 weeks, "
            f"stockout risk {row['Risk_Level']}, reorder {row['Recommended_Reorder']:.0f} units"
        )
        pdf.drawString(50, y_pos, line[:95])
        y_pos -= 14
        if y_pos < 60:
            pdf.showPage()
            y_pos = 750
    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.read(), None
def build_material_summary(weekly_df):
    materials = weekly_df["Material"].unique()
    rows = []
    for material in materials:
        material_history = weekly_df[weekly_df["Material"] == material].sort_values(["Year", "Week"])
        recent_actual = float(material_history["Demand"].tail(4).sum())
        forecast_df, rmse, mape = simple_forecast_material(weekly_df, material, 12)
        forecast_4w = float(forecast_df["Forecast"].head(4).sum())
        growth_pct = ((forecast_4w - recent_actual) / max(recent_actual, 1.0)) * 100
        rows.append(
                {
                    "Material": material,
                "Recent_4W_Actual": recent_actual,
                "Forecast_4W": forecast_4w,
                "Growth_%": growth_pct,
                "RMSE": rmse,
                "MAPE_%": mape,
            }
        )
    return pd.DataFrame(rows).sort_values("Forecast_4W", ascending=False).reset_index(drop=True)
def build_planning_table(weekly_df, stock_map, unit_cost_map, lead_time_weeks, service_buffer_weeks):
    rows = []
    for material in weekly_df["Material"].unique():
        forecast_df, rmse, _ = simple_forecast_material(weekly_df, material, max(lead_time_weeks + service_buffer_weeks, 4))
        avg_weekly_forecast = float(forecast_df["Forecast"].head(max(lead_time_weeks, 1)).mean())
        demand_during_lead = float(forecast_df["Forecast"].head(lead_time_weeks).sum())
        safety_stock = avg_weekly_forecast * service_buffer_weeks + (0.5 * rmse)
        reorder_point = demand_during_lead + safety_stock
        current_stock = float(stock_map[material])
        weeks_cover = current_stock / max(avg_weekly_forecast, 1.0)
        recommended_reorder = max(0.0, reorder_point - current_stock)
        projected_gap = max(0.0, demand_during_lead - current_stock)
        if current_stock < demand_during_lead:
            risk_level = "High"
        elif current_stock < reorder_point:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        rows.append(
                {
                    "Material": material,
                "Current_Stock": current_stock,
                "Lead_Time_Demand": demand_during_lead,
                "Safety_Stock": safety_stock,
                "Reorder_Point": reorder_point,
                "Recommended_Reorder": recommended_reorder,
                "Projected_Gap": projected_gap,
                "Weeks_of_Cover": weeks_cover,
                "Risk_Level": risk_level,
                "Estimated_Reorder_Cost": recommended_reorder * unit_cost_map[material],
            }
        )
    planning_df = pd.DataFrame(rows)
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    planning_df["Risk_Sort"] = planning_df["Risk_Level"].map(risk_order)
    planning_df = planning_df.sort_values(["Risk_Sort", "Projected_Gap"], ascending=[True, False]).drop(columns=["Risk_Sort"])
    return planning_df.reset_index(drop=True)
def render_metric_card(title, value, delta=None):
    delta_html = f"<div class='metric-delta'>{delta}</div>" if delta is not None else ""
    st.markdown(
            f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
@st.cache_data
def load_forecast_data():
    df = create_synthetic_data("2025-08-01", "2026-04-30")
    weekly_df = aggregate_to_weekly(df)
    summary_df = build_material_summary(weekly_df)
    return df, weekly_df, summary_df
# ============================================================================
# STREAMLIT UI
# ============================================================================
st.set_page_config(page_title="PowerGrid Demand Forecasting", layout="wide")
init_auth_db()
st.markdown(
        """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
            background: linear-gradient(180deg, #f4f7fb 0%, #edf2f7 100%);
        color: #10233f;
    }
    [data-testid="stSidebar"] {
            background: #10233f;
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    [data-testid="stSidebar"] * {
            color: #f5f7fb;
    }
    .hero-card, .panel-card, .metric-card {
            background: rgba(255,255,255,0.92);
        border: 1px solid #d8e2ef;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 12px 28px rgba(16, 35, 63, 0.08);
    }
    .hero-card {
            padding: 1.5rem;
        background: linear-gradient(135deg, #10233f 0%, #1f4e79 100%);
        color: #f5f7fb;
        margin-bottom: 1rem;
    }
    .hero-eyebrow {
            text-transform: uppercase;
        letter-spacing: 0.08em;
        font-size: 0.74rem;
        opacity: 0.8;
        margin-bottom: 0.35rem;
    }
    .hero-title {
            font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }
    .hero-copy {
            font-size: 0.96rem;
        opacity: 0.92;
        margin-bottom: 0;
    }
    .metric-title {
            font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #58708b;
        margin-bottom: 0.45rem;
    }
    .metric-value {
            font-size: 1.7rem;
        font-weight: 700;
        color: #10233f;
        line-height: 1.2;
    }
    .metric-delta {
            margin-top: 0.35rem;
        font-size: 0.9rem;
        color: #365f91;
    }
    .section-title {
            font-size: 1.15rem;
        font-weight: 700;
        margin: 0 0 0.8rem 0;
        color: #10233f;
    }
    .small-note {
            color: #5b6f86;
        font-size: 0.9rem;
    }
    .stButton > button, .stDownloadButton > button {
            background: #143a63;
        color: #ffffff;
        border-radius: 10px;
        border: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
with st.sidebar:
    st.markdown("## PowerGrid")
    st.caption("Material Demand Forecasting")
    st.divider()
    if st.session_state.logged_in:
        st.write(f"User: {st.session_state.username}")
        st.caption(st.session_state.user_role or "")
        if st.session_state.login_time:
            st.caption(f"Login time: {st.session_state.login_time.strftime('%Y-%m-%d %H:%M')}")
        st.divider()
        selected_page = st.radio(
                "Navigation",
            ["Overview", "Forecast Explorer", "Planning and Risk", "Reports"],
            index=["Overview", "Forecast Explorer", "Planning and Risk", "Reports"].index(st.session_state.page),
            key="nav_page",
            label_visibility="visible",
        )
        st.session_state.page = selected_page
        st.divider()
        if st.button("Log out", use_container_width=True):
            logout_user()
            st.rerun()
    else:
        st.write("Sign in to access the dashboard.")
if not st.session_state.logged_in:
    render_login_screen()
    st.stop()
page = st.session_state.page
df, weekly_df, summary_df = load_forecast_data()
st.markdown(
        """
    <div class="hero-card">
        <div class="hero-eyebrow">Forecast Center</div>
        <div class="hero-title">Professional Forecasting Workspace</div>
        <p class="hero-copy">
            A simplified view focused on forecast visibility, demand signals, and supply planning action.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
if page == "Overview":
    top_risk_materials = summary_df.sort_values("Growth_%", ascending=False).head(3)
    strongest_material = summary_df.iloc[0]
    total_forecast_4w = summary_df["Forecast_4W"].sum()
    avg_mape = summary_df["MAPE_%"].mean()
    growth_materials = int((summary_df["Growth_%"] > 0).sum())
    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Forecasted demand next 4 weeks", f"{total_forecast_4w:,.0f} units")
    with metric_cols[1]:
        render_metric_card("Highest demand material", strongest_material["Material"], f"{strongest_material['Forecast_4W']:.0f} units")
    with metric_cols[2]:
        render_metric_card("Average model MAPE", f"{avg_mape:.1f}%")
    with metric_cols[3]:
        render_metric_card("Materials with positive growth", f"{growth_materials}")
    chart_col, insight_col = st.columns([1.7, 1])
    with chart_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Material Demand Outlook</div>', unsafe_allow_html=True)
        overview_chart = px.bar(
                summary_df.sort_values("Forecast_4W", ascending=True),
            x="Forecast_4W",
            y="Material",
            orientation="h",
            color="Growth_%",
            color_continuous_scale=["#8fa8c3", "#2d5f8b", "#143a63"],
            labels={"Forecast_4W": "Forecast next 4 weeks", "Material": "Material"},
        )
        overview_chart.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(overview_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with insight_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Priority Signals</div>', unsafe_allow_html=True)
        for _, row in top_risk_materials.iterrows():
            st.markdown(
                    f"""
                <div class="metric-card" style="margin-bottom:0.7rem;">
                    <div class="metric-title">{row['Material']}</div>
                    <div class="metric-value">{row['Forecast_4W']:.0f} units</div>
                    <div class="metric-delta">{row['Growth_%']:+.1f}% vs recent 4 weeks</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    lower_col, upper_col = st.columns([1.1, 1.2])
    with lower_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Demand Trend</div>', unsafe_allow_html=True)
        daily_trend = df.groupby("Date")["Demand"].sum().reset_index()
        trend_chart = px.line(
                daily_trend,
            x="Date",
            y="Demand",
            labels={"Demand": "Daily demand", "Date": "Date"},
        )
        trend_chart.update_traces(line_color="#143a63")
        trend_chart.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(trend_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with upper_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Forecast Performance by Material</div>', unsafe_allow_html=True)
        perf_view = summary_df[["Material", "Forecast_4W", "Growth_%", "MAPE_%"]].copy()
        perf_view["Forecast_4W"] = perf_view["Forecast_4W"].round(0)
        perf_view["Growth_%"] = perf_view["Growth_%"].round(1)
        perf_view["MAPE_%"] = perf_view["MAPE_%"].round(1)
        st.dataframe(perf_view, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
if page == "Forecast Explorer":
    st.markdown('<div class="section-title">Forecast Explorer</div>', unsafe_allow_html=True)
    material_options = list(weekly_df["Material"].unique())
    filter_cols = st.columns([1.3, 1, 1.2])
    with filter_cols[0]:
        selected_materials = st.multiselect(
                "Materials",
            material_options,
            default=[material_options[0]],
        )
    with filter_cols[1]:
        forecast_horizon = st.slider("Forecast horizon (weeks)", 4, 24, 12)
    with filter_cols[2]:
        st.markdown(
                "<p class='small-note'>Use this view to compare recent demand history against the projected requirement for each selected material.</p>",
            unsafe_allow_html=True,
        )
    if not selected_materials:
        st.info("Select at least one material to view its forecast graph.")
    else:
        for selected_material in selected_materials:
            forecast_df, rmse, mape = simple_forecast_material(weekly_df, selected_material, forecast_horizon)
            hist = weekly_df[weekly_df["Material"] == selected_material].sort_values(["Year", "Week"]).copy()
            hist["Date"] = hist.apply(
                    lambda row: datetime.fromisocalendar(int(row["Year"]), int(row["Week"]), 1), axis=1
            )
            st.markdown(
                    f'<div class="section-title">{selected_material} Forecast View</div>',
                unsafe_allow_html=True,
            )
            summary_cols = st.columns(4)
            forecast_total = float(forecast_df["Forecast"].sum())
            recent_actual = float(hist["Demand"].tail(min(4, len(hist))).sum())
            forecast_delta = ((forecast_total - recent_actual) / max(recent_actual, 1.0)) * 100
            with summary_cols[0]:
                render_metric_card("Forecast horizon demand", f"{forecast_total:,.0f} units")
            with summary_cols[1]:
                render_metric_card("Recent 4-week actual", f"{recent_actual:,.0f} units")
            with summary_cols[2]:
                render_metric_card("Expected change", f"{forecast_delta:+.1f}%")
            with summary_cols[3]:
                render_metric_card("Model accuracy", f"MAPE {mape:.1f}%")
            chart_col, side_col = st.columns([1.8, 1])
            with chart_col:
                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Historical vs Forecast</div>', unsafe_allow_html=True)
                fig = go.Figure()
                fig.add_trace(
                        go.Scatter(
                            x=hist["Date"],
                        y=hist["Demand"],
                        mode="lines+markers",
                        name="Historical demand",
                        line=dict(color="#143a63", width=3),
                    )
                )
                fig.add_trace(
                        go.Scatter(
                            x=forecast_df["Date"],
                        y=forecast_df["Upper_Band"],
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                        go.Scatter(
                            x=forecast_df["Date"],
                        y=forecast_df["Lower_Band"],
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(54,95,145,0.18)",
                        line=dict(width=0),
                        name="Confidence band",
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                        go.Scatter(
                            x=forecast_df["Date"],
                        y=forecast_df["Forecast"],
                        mode="lines+markers",
                        name="Forecast",
                        line=dict(color="#2d5f8b", width=3, dash="dash"),
                    )
                )
                fig.update_layout(
                        height=450,
                    margin=dict(l=10, r=10, t=60, b=10),
                    title=dict(
                        text=f"{selected_material} weekly forecast",
                        x=0.02,
                        y=0.98,
                        xanchor="left",
                        yanchor="top",
                    ),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"forecast_chart_{selected_material}")
                st.caption("Historical weekly totals exclude partial start and end weeks to avoid misleading drops.")
                st.markdown("</div>", unsafe_allow_html=True)
            with side_col:
                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Forecast Notes</div>', unsafe_allow_html=True)
                st.write(f"Material: {selected_material}")
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAPE: {mape:.2f}%")
                st.write(f"Forecast horizon: {forecast_horizon} weeks")
                peak_row = forecast_df.loc[forecast_df["Forecast"].idxmax()]
                st.write(f"Peak projected week: {peak_row['Date'].strftime('%d %b %Y')}")
                st.write(f"Peak projected demand: {peak_row['Forecast']:.1f} units")
                st.markdown(
                        "<p class='small-note'>This simplified model uses trend and seasonal weekly offsets. It is suitable for a project prototype and can later be replaced with a stronger forecasting model.</p>",
                    unsafe_allow_html=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Forecast Table</div>', unsafe_allow_html=True)
            display_forecast = forecast_df[
                    ["Date", "Week", "Forecast", "Lower_Band", "Upper_Band", "Bias_vs_Mean_%"]
            ].copy()
            display_forecast["Date"] = display_forecast["Date"].dt.strftime("%Y-%m-%d")
            display_forecast = display_forecast.round(
                    {"Forecast": 1, "Lower_Band": 1, "Upper_Band": 1, "Bias_vs_Mean_%": 1}
            )
            st.dataframe(display_forecast, hide_index=True, use_container_width=True)
            csv_forecast = forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                    f"Download {selected_material} forecast CSV",
                data=csv_forecast,
                file_name=f"forecast_{selected_material}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key=f"forecast_download_{selected_material}",
            )
            st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
if page == "Planning and Risk":
    st.markdown('<div class="section-title">Planning and Risk</div>', unsafe_allow_html=True)
    control_cols = st.columns([1, 1, 1.2])
    with control_cols[0]:
        lead_time_weeks = st.slider("Lead time (weeks)", 1, 12, 4)
    with control_cols[1]:
        service_buffer_weeks = st.slider("Safety buffer (weeks)", 1, 6, 2)
    with control_cols[2]:
        st.markdown(
                "<p class='small-note'>This section converts forecast demand into reorder points, stock coverage, and shortage alerts.</p>",
            unsafe_allow_html=True,
        )
    with st.expander("Edit stock and unit cost assumptions", expanded=False):
        stock_map = {}
        unit_cost_map = {}
        stock_cols = st.columns(3)
        for idx, material in enumerate(weekly_df["Material"].unique()):
            with stock_cols[idx % 3]:
                stock_map[material] = st.number_input(
                        f"{material} stock",
                    min_value=0.0,
                    value=float(DEFAULT_STOCK.get(material, 300)),
                    step=10.0,
                    key=f"stock_{material}",
                )
                unit_cost_map[material] = st.number_input(
                        f"{material} unit cost",
                    min_value=0.0,
                    value=float(DEFAULT_UNIT_COST.get(material, 10.0)),
                    step=0.5,
                    key=f"cost_{material}",
                )
    planning_df = build_planning_table(
            weekly_df,
        stock_map,
        unit_cost_map,
        lead_time_weeks,
        service_buffer_weeks,
    )
    total_reorder_cost = planning_df["Estimated_Reorder_Cost"].sum()
    high_risk_count = int((planning_df["Risk_Level"] == "High").sum())
    medium_risk_count = int((planning_df["Risk_Level"] == "Medium").sum())
    avg_cover = planning_df["Weeks_of_Cover"].mean()
    risk_metrics = st.columns(4)
    with risk_metrics[0]:
        render_metric_card("High risk materials", f"{high_risk_count}")
    with risk_metrics[1]:
        render_metric_card("Medium risk materials", f"{medium_risk_count}")
    with risk_metrics[2]:
        render_metric_card("Average weeks of cover", f"{avg_cover:.1f}")
    with risk_metrics[3]:
        render_metric_card("Estimated reorder cost", f"Rs {total_reorder_cost:,.0f}")
    table_col, action_col = st.columns([1.8, 1])
    with table_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Material Risk Table</div>', unsafe_allow_html=True)
        planning_view = planning_df.copy()
        planning_view = planning_view.round(
                {
                    "Current_Stock": 0,
                "Lead_Time_Demand": 0,
                "Safety_Stock": 0,
                "Reorder_Point": 0,
                "Recommended_Reorder": 0,
                "Projected_Gap": 0,
                "Weeks_of_Cover": 1,
                "Estimated_Reorder_Cost": 0,
            }
        )
        st.dataframe(planning_view, hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with action_col:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Immediate Actions</div>', unsafe_allow_html=True)
        high_risk_df = planning_df[planning_df["Risk_Level"] == "High"].head(5)
        if high_risk_df.empty:
            st.success("No materials are currently in the high-risk band.")
        else:
            for _, row in high_risk_df.iterrows():
                st.markdown(
                        f"""
                    <div class="metric-card" style="margin-bottom:0.7rem;">
                        <div class="metric-title">{row['Material']}</div>
                        <div class="metric-value">{row['Recommended_Reorder']:.0f} units</div>
                        <div class="metric-delta">Projected gap {row['Projected_Gap']:.0f} units</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Reorder Cost by Material</div>', unsafe_allow_html=True)
    cost_chart = px.bar(
            planning_df.sort_values("Estimated_Reorder_Cost", ascending=True),
        x="Estimated_Reorder_Cost",
        y="Material",
        orientation="h",
        color="Risk_Level",
        color_discrete_map={"High": "#b63a3a", "Medium": "#d08a22", "Low": "#2d7d5a"},
        labels={"Estimated_Reorder_Cost": "Estimated reorder cost", "Material": "Material"},
    )
    cost_chart.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(cost_chart, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
if page == "Reports":
    stock_map = {material: float(DEFAULT_STOCK.get(material, 300)) for material in weekly_df["Material"].unique()}
    unit_cost_map = {material: float(DEFAULT_UNIT_COST.get(material, 10.0)) for material in weekly_df["Material"].unique()}
    planning_df = build_planning_table(weekly_df, stock_map, unit_cost_map, lead_time_weeks=4, service_buffer_weeks=2)
    st.markdown('<div class="section-title">Reports and Exports</div>', unsafe_allow_html=True)
    report_cols = st.columns([1.1, 1.4])
    with report_cols[0]:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Management Snapshot</div>', unsafe_allow_html=True)
        st.write(f"Materials tracked: {weekly_df['Material'].nunique()}")
        st.write(f"Forecasted demand next 4 weeks: {summary_df['Forecast_4W'].sum():,.0f} units")
        st.write(f"High risk materials: {(planning_df['Risk_Level'] == 'High').sum()}")
        st.write(f"Average MAPE: {summary_df['MAPE_%'].mean():.1f}%")
        st.markdown(
                "<p class='small-note'>Use this section to export the clean forecasting outputs instead of browsing multiple tabs.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with report_cols[1]:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Download Outputs</div>', unsafe_allow_html=True)
        summary_csv = summary_df.to_csv(index=False).encode("utf-8")
        planning_csv = planning_df.to_csv(index=False).encode("utf-8")
        full_csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
                "Download forecast summary CSV",
            data=summary_csv,
            file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        st.download_button(
                "Download planning and risk CSV",
            data=planning_csv,
            file_name=f"planning_risk_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        st.download_button(
                "Download historical dataset CSV",
            data=full_csv,
            file_name=f"historical_dataset_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        pdf_bytes, err = create_pdf_report_bytes(
                df,
            weekly_df,
            st.session_state.username,
            st.session_state.user_role,
            planning_df[["Material", "Recommended_Reorder", "Risk_Level"]].merge(
                    summary_df[["Material", "Forecast_4W"]],
                on="Material",
                how="left",
            ),
        )
        if err:
            st.warning(f"PDF export unavailable: {err}.")
        else:
            st.download_button(
                    "Download management PDF",
                data=pdf_bytes,
                file_name=f"powergrid_forecast_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
            )
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Forecast Summary Table</div>', unsafe_allow_html=True)
    report_table = summary_df.merge(
            planning_df[["Material", "Risk_Level", "Recommended_Reorder", "Estimated_Reorder_Cost"]],
        on="Material",
        how="left",
    )
    report_table = report_table.round(
            {
                "Recent_4W_Actual": 0,
            "Forecast_4W": 0,
            "Growth_%": 1,
            "RMSE": 1,
            "MAPE_%": 1,
            "Recommended_Reorder": 0,
            "Estimated_Reorder_Cost": 0,
        }
    )
    st.dataframe(report_table, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
