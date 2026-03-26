from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html


def render(ctx: dict, split_filter: str, class_filter: str) -> html.Div:
    _ = (split_filter, class_filter)
    feat_meta = ctx["feature_meta"]
    glcm_preview = ctx.get("glcm_preview", {})

    resnet_dim = feat_meta["resnet50_dim"]
    glcm_dim = feat_meta["glcm_dim"]
    total_dim = feat_meta["feature_dim"]

    df_dim = pd.DataFrame(
        {
            "Nguồn đặc trưng": ["ResNet50", "GLCM"],
            "Số chiều": [resnet_dim, glcm_dim],
        }
    )

    fig_dim = px.bar(
        df_dim,
        x="Nguồn đặc trưng",
        y="Số chiều",
        text="Số chiều",
        color="Nguồn đặc trưng",
        color_discrete_sequence=["#2563eb", "#14b8a6"],
        title="Thành phần vector Hybrid feature",
    )
    fig_dim.update_traces(textposition="outside")
    fig_dim.update_layout(height=460)

    low_var = feat_meta["low_variance_features_pct"]

    df_share = pd.DataFrame(
        {
            "Nguồn": ["ResNet50", "GLCM"],
            "Tỷ trọng": [resnet_dim / total_dim * 100, glcm_dim / total_dim * 100],
        }
    )
    fig_share = px.pie(
        df_share,
        values="Tỷ trọng",
        names="Nguồn",
        hole=0.5,
        color="Nguồn",
        color_discrete_map={"ResNet50": "#2563eb", "GLCM": "#14b8a6"},
        title="Tỷ trọng đóng góp số chiều đặc trưng",
    )
    fig_share.update_layout(height=430)

    fig_hybrid_flow = go.Figure()

    nodes = {
        "input": {"x": 0.0, "y": 0.5, "label": "Ảnh X-ray đầu vào", "color": "#475569", "size": 36},
        "resnet": {"x": 1.0, "y": 0.75, "label": "Nhánh 1: ResNet50", "color": "#2563eb", "size": 34},
        "glcm": {"x": 1.0, "y": 0.25, "label": "Nhánh 2: GLCM", "color": "#0f766e", "size": 34},
        "resnet_vec": {"x": 2.0, "y": 0.75, "label": "Vector ResNet (2048)", "color": "#1d4ed8", "size": 34},
        "glcm_vec": {"x": 2.0, "y": 0.25, "label": "Vector GLCM (6)", "color": "#14b8a6", "size": 34},
        "hybrid": {"x": 3.0, "y": 0.5, "label": "Hybrid Vector (2054)", "color": "#7c3aed", "size": 40},
    }

    edges = [
        ("input", "resnet", "#8aa2bc", 4),
        ("input", "glcm", "#8aa2bc", 4),
        ("resnet", "resnet_vec", "#3b82f6", 6),
        ("glcm", "glcm_vec", "#14b8a6", 3),
        ("resnet_vec", "hybrid", "#3b82f6", 6),
        ("glcm_vec", "hybrid", "#14b8a6", 3),
    ]

    for src, dst, color, width in edges:
        fig_hybrid_flow.add_trace(
            go.Scatter(
                x=[nodes[src]["x"], nodes[dst]["x"]],
                y=[nodes[src]["y"], nodes[dst]["y"]],
                mode="lines",
                line={"color": color, "width": width},
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig_hybrid_flow.add_annotation(
            x=nodes[dst]["x"],
            y=nodes[dst]["y"],
            ax=nodes[src]["x"],
            ay=nodes[src]["y"],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.1,
            arrowwidth=1.0,
            arrowcolor=color,
            text="",
        )

    fig_hybrid_flow.add_trace(
        go.Scatter(
            x=[nodes[k]["x"] for k in nodes],
            y=[nodes[k]["y"] for k in nodes],
            mode="markers+text",
            marker={
                "size": [nodes[k]["size"] for k in nodes],
                "color": [nodes[k]["color"] for k in nodes],
                "line": {"color": "white", "width": 2},
            },
            text=[nodes[k]["label"] for k in nodes],
            textposition="bottom center",
            textfont={"size": 12, "color": "#1e3a5f"},
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig_hybrid_flow.update_layout(
        title="Sơ đồ 2 nhánh trích xuất đặc trưng và ghép Hybrid",
        height=430,
        margin={"l": 30, "r": 30, "t": 60, "b": 60},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis={"visible": False, "range": [-0.2, 3.2]},
        yaxis={"visible": False, "range": [0.05, 0.95]},
    )

    glcm_cards = []
    if glcm_preview:
        glcm_cards = [
            html.Div([html.Img(src=glcm_preview["original"], className="preview-img"), html.Div("Ảnh X-ray mẫu", className="preview-label")], className="preview-card"),
            html.Div([html.Img(src=glcm_preview["contrast"], className="preview-img"), html.Div("Bản đồ Contrast (vùng biến thiên mạnh)", className="preview-label")], className="preview-card"),
            html.Div([html.Img(src=glcm_preview["homogeneity"], className="preview-img"), html.Div("Bản đồ Homogeneity (vùng đồng nhất)", className="preview-label")], className="preview-card"),
        ]

    return html.Div(
        children=[
            html.H3("Bước 4: Trích xuất đặc trưng", className="step-title"),
            html.P(
                "Mục tiêu: biến ảnh thành vector số giàu thông tin để thuật toán ML phân loại.",
                className="step-desc",
            ),
            html.Div(
                className="insight-grid",
                children=[
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Deep feature (ResNet50)"),
                            html.P(
                                "Backbone pretrained học tốt đặc trưng hình thái toàn cục như vùng mờ lan tỏa, đường biên phổi, và cấu trúc mô bất thường."
                            ),
                        ],
                    ),
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Texture feature (GLCM)"),
                            html.P(
                                "GLCM bổ sung thống kê vi cấu trúc như contrast, homogeneity, correlation giúp mô hình nhạy hơn với biến đổi hạt mịn trong nhu mô phổi."
                            ),
                        ],
                    ),
                ],
            ),
            html.Ul(
                className="step-list",
                children=[
                    html.Li("ResNet50 frozen trích xuất 2048 đặc trưng hình thái."),
                    html.Li("GLCM bổ sung 6 đặc trưng texture vi mô."),
                    html.Li(f"Vector hybrid cuối cùng: {total_dim} chiều."),
                    html.Li(f"Tỷ lệ feature low-variance: {low_var:.2f}% (thấp, chất lượng tốt)."),
                ],
            ),
            html.Div(className="grid-2", children=[dcc.Graph(figure=fig_dim), dcc.Graph(figure=fig_share)]),
            dcc.Graph(figure=fig_hybrid_flow),
            html.H4("Minh họa trực quan đặc trưng GLCM trên ảnh X-quang", style={"marginTop": "6px", "color": "#115084"}),
            html.P(
                "Contrast cao thường xuất hiện ở vùng biên hoặc vùng có thay đổi cường độ mạnh; Homogeneity cao thường nằm ở vùng nền đồng đều.",
                className="step-desc",
            ),
            html.Div(className="preview-grid", children=glcm_cards or [html.Div("Không tìm thấy ảnh mẫu để minh họa GLCM.")]),
            html.Div(
                className="kpi-row",
                children=[
                    html.Div(
                        [
                            html.Div("GLCM Contrast (global)", className="kpi-label"),
                            html.Div(f"{glcm_preview.get('contrast_value', 0.0):.3f}", className="kpi-value"),
                        ],
                        className="kpi-card",
                    ),
                    html.Div(
                        [
                            html.Div("GLCM Homogeneity (global)", className="kpi-label"),
                            html.Div(f"{glcm_preview.get('homogeneity_value', 0.0):.3f}", className="kpi-value"),
                        ],
                        className="kpi-card",
                    ),
                ],
            ),
            html.Div(
                className="formula-box",
                children=[
                    html.Strong("Hybrid feature vector:"),
                    html.Span(
                        f" X = [X_resnet, X_glcm] với tổng {total_dim} chiều, sau đó đưa qua StandardScaler và PCA trước khi train Stacking."
                    ),
                ],
            ),
        ]
    )
