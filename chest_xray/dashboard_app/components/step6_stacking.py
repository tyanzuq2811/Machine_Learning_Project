from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import dcc, html, dash_table


def render(ctx: dict, split_filter: str, class_filter: str) -> html.Div:
    _ = (split_filter, class_filter)
    result_meta = ctx["result_meta"]

    df = pd.DataFrame(
        [
            {"Model": "SVM", **result_meta["individual_models"]["svm"]},
            {"Model": "Random Forest", **result_meta["individual_models"]["rf"]},
            {"Model": "XGBoost", **result_meta["individual_models"]["xgb"]},
            {
                "Model": "Stacking",
                "accuracy": result_meta["stacking_ensemble"]["test_accuracy"],
                "precision": result_meta["stacking_ensemble"]["test_precision"],
                "recall": result_meta["stacking_ensemble"]["test_recall"],
                "f1": result_meta["stacking_ensemble"]["test_f1"],
            },
        ]
    )

    df_long = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    df_long["Score"] *= 100

    fig = px.bar(
        df_long,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text="Score",
        title="So sánh model trong Stacking Ensemble",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_yaxes(range=[80, 101])
    fig.update_layout(height=490)

    best_base_acc = max(result_meta["individual_models"][m]["accuracy"] for m in ["svm", "rf", "xgb"])
    stacking_acc = result_meta["stacking_ensemble"]["test_accuracy"]
    gain_pct = (stacking_acc - best_base_acc) * 100

    table_df = df.copy()
    for c in ["accuracy", "precision", "recall", "f1"]:
        table_df[c] = (table_df[c] * 100).map(lambda x: f"{x:.2f}%")

    return html.Div(
        children=[
            html.H3("Bước 6: Mô hình Stacking Ensemble", className="step-title"),
            html.P(
                "Mục tiêu: kết hợp điểm mạnh của SVM, Random Forest, XGBoost và meta-model Logistic Regression.",
                className="step-desc",
            ),
            html.Div(
                className="insight-grid",
                children=[
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Vai trò tầng base"),
                            html.P(
                                "Mỗi mô hình base học một góc nhìn khác nhau trên cùng feature space: SVM mạnh biên phân tách, RF/XGB mạnh quan hệ phi tuyến."
                            ),
                        ],
                    ),
                    html.Div(
                        className="insight-card",
                        children=[
                            html.H4("Vai trò tầng meta"),
                            html.P(
                                "Logistic Regression học cách phối hợp xác suất từ các base model để đưa ra quyết định cuối ổn định hơn từng mô hình riêng lẻ."
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="kpi-row",
                children=[
                    html.Div([html.Div("Best base Accuracy", className="kpi-label"), html.Div(f"{best_base_acc * 100:.2f}%", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("Stacking Accuracy", className="kpi-label"), html.Div(f"{stacking_acc * 100:.2f}%", className="kpi-value")], className="kpi-card"),
                    html.Div([html.Div("Mức cải thiện", className="kpi-label"), html.Div(f"+{gain_pct:.2f} điểm", className="kpi-value")], className="kpi-card"),
                ],
            ),
            html.Ul(
                className="step-list",
                children=[
                    html.Li("Tầng 0: SVM (RBF) + RF + XGB."),
                    html.Li("Tầng 1 (meta): Logistic Regression học cách phối hợp 3 đầu ra."),
                    html.Li("Kết quả cuối là xác suất và nhãn NORMAL/PNEUMONIA."),
                ],
            ),
            dcc.Graph(figure=fig),
            dash_table.DataTable(
                columns=[
                    {"name": "Model", "id": "Model"},
                    {"name": "Accuracy", "id": "accuracy"},
                    {"name": "Precision", "id": "precision"},
                    {"name": "Recall", "id": "recall"},
                    {"name": "F1", "id": "f1"},
                ],
                data=table_df.to_dict("records"),
                style_table={"overflowX": "auto"},
                style_header={"fontWeight": "700", "backgroundColor": "#e2e8f0"},
                style_cell={"padding": "10px", "textAlign": "center", "minWidth": "120px"},
            ),
        ]
    )
