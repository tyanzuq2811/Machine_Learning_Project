from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import dcc, html


def render(ctx: dict, split_filter: str, class_filter: str) -> html.Div:
    _ = (split_filter, class_filter)
    split_meta = ctx["split_meta"]
    feat_meta = ctx["feature_meta"]
    preview = ctx.get("augmentation_preview", {})

    before_train = split_meta["splits"]["train"]["total"]
    after_train = feat_meta["shapes"]["train"][0]

    df_aug = pd.DataFrame(
        {
            "Stage": ["Train trước augmentation", "Train sau augmentation"],
            "Images": [before_train, after_train],
        }
    )

    fig_aug = px.bar(
        df_aug,
        x="Stage",
        y="Images",
        text="Images",
        color="Stage",
        color_discrete_sequence=["#0ea5e9", "#6366f1"],
        title="Tác động của augmentation lên tập Train",
    )
    fig_aug.update_traces(textposition="outside")
    fig_aug.update_layout(height=470)

    growth_pct = (after_train - before_train) / before_train * 100 if before_train else 0.0

    cards = []
    if preview:
        cards = [
            html.Div([html.Img(src=preview["original"], className="preview-img"), html.Div("Gốc", className="preview-label")], className="preview-card"),
            html.Div([html.Img(src=preview["flipped"], className="preview-img"), html.Div("Lật ngang", className="preview-label")], className="preview-card"),
            html.Div([html.Img(src=preview["rotated"], className="preview-img"), html.Div("Xoay +10°", className="preview-label")], className="preview-card"),
        ]

    return html.Div(
        children=[
            html.H3("Bước 3: Tăng cường dữ liệu (Augmentation)", className="step-title"),
            html.P(
                "Mục tiêu: tăng tính đa dạng dữ liệu train để mô hình khái quát tốt hơn và giảm overfitting.",
                className="step-desc",
            ),
            html.Div(
                className="insight-grid",
                children=[
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Tại sao chỉ augment train"),
                            html.P(
                                "Validation/Test đại diện cho dữ liệu thật chưa thấy. Nếu augment cả các tập này, kết quả đánh giá sẽ bị sai lệch và quá lạc quan."
                            ),
                        ],
                    ),
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Mức độ biến đổi hợp lý"),
                            html.P(
                                "Biên độ xoay nhỏ (±15°) mô phỏng sai khác tư thế chụp thực tế, nhưng vẫn bảo toàn cấu trúc giải phẫu quan trọng để chẩn đoán."
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="kpi-row",
                children=[
                    html.Div([html.Div("Train trước", className="kpi-label"), html.Div(f"{before_train:,}", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("Train sau", className="kpi-label"), html.Div(f"{after_train:,}", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("Mức tăng", className="kpi-label"), html.Div(f"+{growth_pct:.1f}%", className="kpi-value")], className="kpi-card"),
                ],
            ),
            html.Ul(
                className="step-list",
                children=[
                    html.Li("Áp dụng trên train: horizontal flip + random rotation ±15°."),
                    html.Li("Không áp dụng trên val/test để đảm bảo đánh giá khách quan."),
                    html.Li("Không dùng vertical flip vì không phù hợp giải phẫu ảnh X-quang phổi."),
                    html.Li(f"Train tăng từ {before_train:,} lên {after_train:,} ảnh."),
                ],
            ),
            dcc.Graph(figure=fig_aug),
            html.Div(className="preview-grid", children=cards or [html.Div("Không tìm thấy ảnh mẫu augmentation.")]),
        ]
    )
