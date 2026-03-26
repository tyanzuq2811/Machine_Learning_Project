from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html


def render(ctx: dict, split_filter: str, class_filter: str) -> html.Div:
    _ = (split_filter, class_filter)
    preview = ctx.get("preprocess_preview", {})

    step_labels = ["Ảnh X-ray gốc", "CLAHE", "Resize 224x224", "3-channel", "Normalize ImageNet"]
    x_nodes = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_nodes = [0.5] * len(x_nodes)
    node_colors = ["#64748b", "#0ea5e9", "#7c3aed", "#0891b2", "#16a34a"]

    fig_flow = go.Figure()

    for i in range(len(x_nodes) - 1):
        fig_flow.add_trace(
            go.Scatter(
                x=[x_nodes[i], x_nodes[i + 1]],
                y=[y_nodes[i], y_nodes[i + 1]],
                mode="lines",
                line={"color": "#9db6cf", "width": 5},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig_flow.add_trace(
        go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode="markers+text",
            marker={"size": 42, "color": node_colors, "line": {"color": "white", "width": 2}},
            text=step_labels,
            textposition="bottom center",
            textfont={"size": 12, "color": "#1e3a5f"},
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    for i in range(len(x_nodes) - 1):
        fig_flow.add_annotation(
            x=(x_nodes[i] + x_nodes[i + 1]) / 2,
            y=0.5,
            ax=x_nodes[i],
            ay=0.5,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1.2,
            arrowwidth=1.2,
            arrowcolor="#6f8fb1",
            text="",
        )

    fig_flow.update_layout(
        title="Luồng tiền xử lý ảnh",
        height=430,
        margin={"l": 30, "r": 30, "t": 60, "b": 60},
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis={"visible": False, "range": [-0.3, 4.3]},
        yaxis={"visible": False, "range": [0.2, 0.8]},
    )

    preview_cards = []
    if preview:
        preview_cards = [
            html.Div([html.Img(src=preview["original"], className="preview-img"), html.Div("Gốc", className="preview-label")], className="preview-card"),
            html.Div([html.Img(src=preview["clahe"], className="preview-img"), html.Div("Sau CLAHE", className="preview-label")], className="preview-card"),
            html.Div([html.Img(src=preview["resized"], className="preview-img"), html.Div("Resize 224x224", className="preview-label")], className="preview-card"),
        ]

    return html.Div(
        children=[
            html.H3("Bước 2: Tiền xử lý ảnh", className="step-title"),
            html.P(
                "Mục tiêu: đưa ảnh về chuẩn chung để ResNet50 xử lý ổn định, giảm nhiễu từ chênh lệch máy chụp.",
                className="step-desc",
            ),
            html.Div(
                className="insight-grid",
                children=[
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Chuẩn hóa cường độ sáng"),
                            html.P(
                                "CLAHE giúp mở rộng độ tương phản ở từng vùng nhỏ trong phổi, hữu ích khi tổn thương có độ mờ nhẹ và khó thấy bằng histogram toàn ảnh."
                            ),
                        ],
                    ),
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Chuẩn hóa kích thước đầu vào"),
                            html.P(
                                "Tất cả ảnh được đưa về 224x224 để giữ tương thích với backbone ResNet50 pretrained, đồng thời đảm bảo batch inference ổn định."
                            ),
                        ],
                    ),
                ],
            ),
            html.Ul(
                className="step-list",
                children=[
                    html.Li("CLAHE tăng tương phản cục bộ, làm rõ vùng mờ do viêm."),
                    html.Li("Resize toàn bộ ảnh về 224x224 với nội suy LANCZOS."),
                    html.Li("Chuyển grayscale sang 3 kênh để tương thích ResNet50 pretrained."),
                    html.Li("Normalize theo chuẩn ImageNet bằng preprocess_input."),
                ],
            ),
            dcc.Graph(figure=fig_flow),
            html.Div(className="preview-grid", children=preview_cards or [html.Div("Không tìm thấy ảnh mẫu để preview.")]),
            html.Div(
                className="formula-box",
                children=[
                    html.Strong("Ghi chú kỹ thuật:"),
                    html.Span(
                        " Pipeline inference phải dùng đúng thứ tự tiền xử lý giống giai đoạn train để tránh lệch phân phối dữ liệu đầu vào."
                    ),
                ],
            ),
        ]
    )
