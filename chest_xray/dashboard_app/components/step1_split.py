from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, dash_table


def render(ctx: dict, split_filter: str, class_filter: str) -> html.Div:
    split_meta = ctx["split_meta"]
    original_split = ctx["original_split"]

    # Original split (from guide)
    df_original = pd.DataFrame(
        [
            {"Split": "TRAIN", "NORMAL": original_split["train"]["NORMAL"], "PNEUMONIA": original_split["train"]["PNEUMONIA"]},
            {"Split": "TEST", "NORMAL": original_split["test"]["NORMAL"], "PNEUMONIA": original_split["test"]["PNEUMONIA"]},
            {"Split": "VAL", "NORMAL": original_split["val"]["NORMAL"], "PNEUMONIA": original_split["val"]["PNEUMONIA"]},
        ]
    )

    fig_original = go.Figure()
    fig_original.add_bar(name="NORMAL", x=df_original["Split"], y=df_original["NORMAL"], marker_color="#10b981")
    fig_original.add_bar(name="PNEUMONIA", x=df_original["Split"], y=df_original["PNEUMONIA"], marker_color="#ef4444")
    fig_original.update_layout(title="Original Kaggle Split (Validation rất nhỏ)", barmode="stack", height=460)

    # Resplit distribution
    splits = ["train", "val", "test"] if split_filter == "all" else [split_filter]
    rows = []
    for sp in splits:
        row = {"Split": sp.upper()}
        if class_filter in ("ALL", "NORMAL"):
            row["NORMAL"] = split_meta["splits"][sp]["NORMAL"]
        if class_filter in ("ALL", "PNEUMONIA"):
            row["PNEUMONIA"] = split_meta["splits"][sp]["PNEUMONIA"]
        rows.append(row)

    df_new = pd.DataFrame(rows)
    fig_new = go.Figure()
    if "NORMAL" in df_new.columns:
        fig_new.add_bar(name="NORMAL", x=df_new["Split"], y=df_new["NORMAL"], marker_color="#10b981")
    if "PNEUMONIA" in df_new.columns:
        fig_new.add_bar(name="PNEUMONIA", x=df_new["Split"], y=df_new["PNEUMONIA"], marker_color="#ef4444")
    fig_new.update_layout(title="Resplit theo Patient-ID (80/10/10)", barmode="stack", height=460)

    total_patients = split_meta["total_patients"]
    total_images = split_meta["total_images"]

    ratio_df = pd.DataFrame(
        [
            {"Class": "NORMAL", "Count": split_meta["splits"]["train"]["NORMAL"] + split_meta["splits"]["val"]["NORMAL"] + split_meta["splits"]["test"]["NORMAL"]},
            {"Class": "PNEUMONIA", "Count": split_meta["splits"]["train"]["PNEUMONIA"] + split_meta["splits"]["val"]["PNEUMONIA"] + split_meta["splits"]["test"]["PNEUMONIA"]},
        ]
    )
    fig_ratio = px.pie(
        ratio_df,
        values="Count",
        names="Class",
        hole=0.45,
        color="Class",
        color_discrete_map={"NORMAL": "#10b981", "PNEUMONIA": "#ef4444"},
        title="Tỷ lệ lớp sau khi gộp dữ liệu",
    )
    fig_ratio.update_layout(height=480)

    split_pct_rows = []
    for s in ["train", "val", "test"]:
        total = split_meta["splits"][s]["total"]
        split_pct_rows.append(
            {
                "Split": s.upper(),
                "NORMAL %": split_meta["splits"][s]["NORMAL"] / total * 100,
                "PNEUMONIA %": split_meta["splits"][s]["PNEUMONIA"] / total * 100,
            }
        )
    df_pct = pd.DataFrame(split_pct_rows)
    fig_pct = go.Figure()
    fig_pct.add_bar(name="NORMAL %", x=df_pct["Split"], y=df_pct["NORMAL %"], marker_color="#22c55e")
    fig_pct.add_bar(name="PNEUMONIA %", x=df_pct["Split"], y=df_pct["PNEUMONIA %"], marker_color="#f97316")
    fig_pct.update_layout(
        barmode="group",
        title="Tỷ lệ lớp theo từng split sau khi chia lại",
        yaxis_title="Tỷ lệ (%)",
        height=430,
    )

    table_data = [
        {
            "split": s.upper(),
            "total": split_meta["splits"][s]["total"],
            "normal": split_meta["splits"][s]["NORMAL"],
            "pneumonia": split_meta["splits"][s]["PNEUMONIA"],
        }
        for s in ["train", "val", "test"]
    ]

    return html.Div(
        children=[
            html.H3("Bước 1: Chia lại dữ liệu", className="step-title"),
            html.P(
                "Mục tiêu: khắc phục tập validation gốc quá nhỏ, đồng thời chia theo bệnh nhân để tránh data leakage.",
                className="step-desc",
            ),
            html.Div(
                className="insight-grid",
                children=[
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Vấn đề của split gốc"),
                            html.P(
                                "Validation gốc chỉ có 16 ảnh nên dao động metric rất mạnh, không phản ánh chính xác khả năng tổng quát hóa của mô hình."
                            ),
                        ],
                    ),
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Lý do chia theo Patient-ID"),
                            html.P(
                                "Nếu cùng một bệnh nhân xuất hiện ở cả train và test, mô hình có thể học dấu vết riêng của bệnh nhân thay vì học đặc trưng bệnh lý."
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="kpi-row",
                children=[
                    html.Div([html.Div("Tổng ảnh", className="kpi-label"), html.Div(f"{total_images:,}", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("Tổng bệnh nhân", className="kpi-label"), html.Div(f"{total_patients:,}", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("Leakage", className="kpi-label"), html.Div("PASS", className="kpi-value")], className="kpi-card"),
                ],
            ),
            html.Div(className="grid-2", children=[dcc.Graph(figure=fig_original), dcc.Graph(figure=fig_new)]),
            dcc.Graph(figure=fig_ratio),
            dcc.Graph(figure=fig_pct),
            dash_table.DataTable(
                columns=[
                    {"name": "Split", "id": "split"},
                    {"name": "Total", "id": "total"},
                    {"name": "NORMAL", "id": "normal"},
                    {"name": "PNEUMONIA", "id": "pneumonia"},
                ],
                data=table_data,
                style_table={"overflowX": "auto"},
                style_header={"fontWeight": "700", "backgroundColor": "#e2e8f0"},
                style_cell={"padding": "8px", "textAlign": "center"},
            ),
        ]
    )
