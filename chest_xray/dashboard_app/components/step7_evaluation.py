from __future__ import annotations

import plotly.graph_objects as go
from dash import dcc, html


METRIC_IMPORTANCE = [
    ("Recall (Sensitivity)", "Quan trọng nhất trong y tế: giảm bỏ sót ca bệnh"),
    ("AUC-ROC", "Đánh giá khả năng phân biệt hai lớp tổng quát"),
    ("F1-Score", "Cân bằng giữa Precision và Recall"),
    ("Specificity", "Khả năng nhận diện đúng người khỏe"),
    ("Precision", "Giảm cảnh báo nhầm"),
    ("Accuracy", "Chỉ số tổng quát, dễ bị ảnh hưởng imbalance"),
]


def render(ctx: dict, split_filter: str, class_filter: str) -> html.Div:
    _ = (split_filter, class_filter)
    result_meta = ctx["result_meta"]
    se = result_meta["stacking_ensemble"]
    cm = result_meta["confusion_matrix"]

    z = [[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]
    fig_cm = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Pred NORMAL", "Pred PNEUMONIA"],
            y=["True NORMAL", "True PNEUMONIA"],
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",
        )
    )
    fig_cm.update_layout(title="Confusion Matrix", height=450)

    fig_gap = go.Figure(
        data=[
            go.Indicator(
                mode="gauge+number",
                value=se["overfitting_pct"],
                number={"suffix": "%"},
                title={"text": "Overfitting Gap (Val - Test)"},
                gauge={
                    "axis": {"range": [0, 10]},
                    "bar": {"color": "#0ea5e9"},
                    "steps": [
                        {"range": [0, 2], "color": "#dcfce7"},
                        {"range": [2, 5], "color": "#fef9c3"},
                        {"range": [5, 10], "color": "#fee2e2"},
                    ],
                },
            )
        ]
    )
    fig_gap.update_layout(height=450)

    fig_err = go.Figure(
        data=[
            go.Bar(
                x=["False Positive", "False Negative"],
                y=[cm["FP"], cm["FN"]],
                marker_color=["#f97316", "#ef4444"],
                text=[cm["FP"], cm["FN"]],
                textposition="outside",
            )
        ]
    )
    fig_err.update_layout(title="Phân tích lỗi quan trọng", yaxis_title="Số lượng", height=420)

    kpis = [
        ("Accuracy", f"{se['test_accuracy'] * 100:.2f}%"),
        ("Precision", f"{se['test_precision'] * 100:.2f}%"),
        ("Recall", f"{se['test_recall'] * 100:.2f}%"),
        ("F1", f"{se['test_f1'] * 100:.2f}%"),
        ("AUC-ROC", f"{se['test_auc_roc']:.4f}"),
        ("Specificity", f"{se['specificity'] * 100:.2f}%"),
    ]

    return html.Div(
        children=[
            html.H3("Bước 7: Đánh giá mô hình", className="step-title"),
            html.P(
                "Mục tiêu: kiểm tra chất lượng dự đoán trên test set và đánh giá mức overfitting.",
                className="step-desc",
            ),
            html.Div(
                className="insight-grid",
                children=[
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Trọng tâm y khoa"),
                            html.P(
                                "Trong bài toán sàng lọc viêm phổi, giảm False Negative là ưu tiên cao nhất vì bỏ sót ca bệnh gây rủi ro lâm sàng lớn."
                            ),
                        ],
                    ),
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Kiểm soát overfitting"),
                            html.P(
                                "Khoảng cách Val-Test thấp cho thấy pipeline từ preprocessing đến Stacking có khả năng tổng quát hóa tốt trên dữ liệu chưa thấy."
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="kpi-row",
                children=[
                    html.Div([html.Div(name, className="kpi-label"), html.Div(val, className="kpi-value")], className="kpi-card")
                    for name, val in kpis
                ],
            ),
            html.Div(className="grid-2", children=[dcc.Graph(figure=fig_cm), dcc.Graph(figure=fig_gap)]),
            dcc.Graph(figure=fig_err),
            html.H4("Thứ tự ưu tiên chỉ số theo góc nhìn y tế", style={"marginTop": "8px"}),
            html.Ol(children=[html.Li(f"{metric}: {desc}") for metric, desc in METRIC_IMPORTANCE], className="step-list"),
            html.P(
                f"Kết luận: model đạt overfitting gap {se['overfitting_pct']:.2f}% (rất thấp), cho thấy khả năng khái quát tốt.",
                className="step-desc",
            ),
        ]
    )
