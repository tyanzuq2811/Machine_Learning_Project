from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import dcc, html


def render(ctx: dict, split_filter: str, class_filter: str) -> html.Div:
    _ = (split_filter, class_filter)
    result_meta = ctx["result_meta"]

    original_dim = result_meta["pca"]["original_dim"]
    reduced_dim = result_meta["pca"]["reduced_dim"]
    variance_kept = result_meta["pca"]["variance_explained"] * 100

    fig_funnel = go.Figure(
        data=[
            go.Funnel(
                y=["Hybrid feature", "Sau PCA"],
                x=[original_dim, reduced_dim],
                textinfo="value+percent initial",
                marker={"color": ["#1d4ed8", "#7c3aed"]},
            )
        ]
    )
    fig_funnel.update_layout(title="Giảm chiều bằng PCA", height=440)

    fig_gauge = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number",
                value=variance_kept,
                number={"suffix": "%"},
                title={"text": "Variance giữ lại"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#0ea5e9"},
                    "steps": [
                        {"range": [0, 80], "color": "#fee2e2"},
                        {"range": [80, 95], "color": "#fef9c3"},
                        {"range": [95, 100], "color": "#dcfce7"},
                    ],
                },
            )
        ]
    )
    fig_gauge.update_layout(height=440)

    df_dim_compare = pd.DataFrame(
        {
            "Giai đoạn": ["Trước PCA", "Sau PCA"],
            "Số chiều": [original_dim, reduced_dim],
        }
    )
    fig_dim_compare = px.bar(
        df_dim_compare,
        x="Giai đoạn",
        y="Số chiều",
        text="Số chiều",
        color="Giai đoạn",
        color_discrete_sequence=["#1d4ed8", "#7c3aed"],
        title="So sánh số chiều đặc trưng trước và sau PCA",
    )
    fig_dim_compare.update_traces(textposition="outside")
    fig_dim_compare.update_layout(height=460, showlegend=False)

    compression = (1 - reduced_dim / original_dim) * 100 if original_dim else 0.0

    return html.Div(
        children=[
            html.H3("Bước 5: StandardScaler + PCA", className="step-title"),
            html.P(
                "Mục tiêu: chuẩn hóa thang đo feature và nén chiều để tăng tốc train, giảm nhiễu.",
                className="step-desc",
            ),
            html.Div(
                className="insight-grid",
                children=[
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Vì sao cần StandardScaler"),
                            html.P(
                                "Các đặc trưng có đơn vị và dải giá trị khác nhau. Chuẩn hóa đưa về cùng thang đo để tránh feature lớn lấn át feature nhỏ."
                            ),
                        ],
                    ),
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Ý nghĩa của PCA 95%"),
                            html.P(
                                "Giữ lại phần lớn phương sai quan trọng, loại bỏ thành phần dư thừa và nhiễu, từ đó giảm nguy cơ overfitting."
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="kpi-row",
                children=[
                    html.Div([html.Div("Chiều ban đầu", className="kpi-label"), html.Div(str(original_dim), className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("Chiều sau PCA", className="kpi-label"), html.Div(str(reduced_dim), className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("Tỷ lệ nén", className="kpi-label"), html.Div(f"{compression:.1f}%", className="kpi-value")], className="kpi-card"),
                ],
            ),
            html.Ul(
                className="step-list",
                children=[
                    html.Li("StandardScaler fit trên train, transform val/test để tránh leakage."),
                    html.Li("PCA giữ 95% thông tin quan trọng."),
                    html.Li(f"Giảm chiều từ {original_dim} xuống {reduced_dim}."),
                ],
            ),
            html.Div(className="grid-2", children=[dcc.Graph(figure=fig_funnel), dcc.Graph(figure=fig_gauge)]),
            dcc.Graph(figure=fig_dim_compare),
            html.Div(
                className="formula-box",
                children=[
                    html.Strong("Nguyên lý PCA:"),
                    html.Span(" tìm các trục trực giao tối đa hóa phương sai và chiếu dữ liệu vào k thành phần chính quan trọng nhất."),
                ],
            ),
        ]
    )
