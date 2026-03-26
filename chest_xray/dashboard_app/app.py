from __future__ import annotations

import os
import sys
from pathlib import Path

from dash import Dash, Input, Output, ctx as dash_ctx, dcc, html
import plotly.io as pio

try:
    # Run as package: python -m chest_xray.dashboard_app.app
    from .components import (
        step1_split,
        step2_preprocess,
        step3_augmentation,
        step4_features,
        step5_pca,
        step6_stacking,
        step7_evaluation,
        step_samples,
    )
    from .data_loader import build_context
except ImportError:
    # Run as script: python chest_xray/dashboard_app/app.py
    CURRENT_DIR = Path(__file__).resolve().parent
    PACKAGE_PARENT = CURRENT_DIR.parent
    if str(PACKAGE_PARENT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_PARENT))

    from dashboard_app.components import (  # type: ignore
        step1_split,
        step2_preprocess,
        step3_augmentation,
        step4_features,
        step5_pca,
        step6_stacking,
        step7_evaluation,
        step_samples,
    )
    from dashboard_app.data_loader import build_context  # type: ignore


STEP_LABELS = {
    "step1": "Bước 1: Chia lại dữ liệu",
    "step2": "Bước 2: Tiền xử lý ảnh",
    "step3": "Bước 3: Tăng cường dữ liệu",
    "step4": "Bước 4: Trích xuất đặc trưng",
    "step5": "Bước 5: Chuẩn hóa + PCA",
    "step6": "Bước 6: Stacking Ensemble",
    "step7": "Bước 7: Đánh giá mô hình",
}


def create_app() -> Dash:
    ctx = build_context()
    pio.templates.default = "plotly_white"
    app = Dash(__name__, suppress_callback_exceptions=True, update_title=None)
    app.title = "EDA Dashboard - 7-Step Pipeline"

    # Pre-render static pages once to avoid expensive re-render on each menu switch.
    static_pages = {
        "step2": step2_preprocess.render(ctx, "all", "ALL"),
        "step3": step3_augmentation.render(ctx, "all", "ALL"),
        "step4": step4_features.render(ctx, "all", "ALL"),
        "step5": step5_pca.render(ctx, "all", "ALL"),
        "step6": step6_stacking.render(ctx, "all", "ALL"),
        "step7": step7_evaluation.render(ctx, "all", "ALL"),
    }

    app.layout = html.Div(
        className="app-shell",
        children=[
            dcc.Store(id="active-page", data="step1"),
            html.Div(
                className="sidebar",
                children=[
                    html.H2("EDA Dashboard", className="sidebar-title"),
                    html.P("Pipeline trực quan 7 bước", className="sidebar-subtitle"),
                    html.Label("Filter split", className="filter-label"),
                    dcc.Dropdown(
                        id="filter-split",
                        options=[
                            {"label": "All", "value": "all"},
                            {"label": "Train", "value": "train"},
                            {"label": "Val", "value": "val"},
                            {"label": "Test", "value": "test"},
                        ],
                        value="all",
                        clearable=False,
                    ),
                    html.Label("Filter class", className="filter-label"),
                    dcc.Dropdown(
                        id="filter-class",
                        options=[
                            {"label": "All", "value": "ALL"},
                            {"label": "NORMAL", "value": "NORMAL"},
                            {"label": "PNEUMONIA", "value": "PNEUMONIA"},
                        ],
                        value="ALL",
                        clearable=False,
                    ),
                    html.Label("Số ảnh mẫu", className="filter-label"),
                    dcc.Slider(id="sample-limit", min=6, max=24, step=2, value=12, marks={6: "6", 12: "12", 24: "24"}),
                    html.Hr(),
                    html.Div(
                        className="menu-list",
                        children=[
                            html.Button(STEP_LABELS["step1"], id="menu-step1", n_clicks=0, className="menu-btn"),
                            html.Button(STEP_LABELS["step2"], id="menu-step2", n_clicks=0, className="menu-btn"),
                            html.Button(STEP_LABELS["step3"], id="menu-step3", n_clicks=0, className="menu-btn"),
                            html.Button(STEP_LABELS["step4"], id="menu-step4", n_clicks=0, className="menu-btn"),
                            html.Button(STEP_LABELS["step5"], id="menu-step5", n_clicks=0, className="menu-btn"),
                            html.Button(STEP_LABELS["step6"], id="menu-step6", n_clicks=0, className="menu-btn"),
                            html.Button(STEP_LABELS["step7"], id="menu-step7", n_clicks=0, className="menu-btn"),
                            html.Button("Tab ảnh mẫu", id="menu-samples", n_clicks=0, className="menu-btn menu-btn-alt"),
                        ],
                    ),
                ],
            ),
            html.Div(
                className="content",
                children=[
                    html.Div(id="menu-header", className="content-header"),
                    dcc.Loading(
                        id="page-loading",
                        type="dot",
                        children=html.Div(id="page-content", className="content-body"),
                    ),
                ],
            ),
        ],
    )

    @app.callback(
        Output("active-page", "data"),
        Input("menu-step1", "n_clicks"),
        Input("menu-step2", "n_clicks"),
        Input("menu-step3", "n_clicks"),
        Input("menu-step4", "n_clicks"),
        Input("menu-step5", "n_clicks"),
        Input("menu-step6", "n_clicks"),
        Input("menu-step7", "n_clicks"),
        Input("menu-samples", "n_clicks"),
        prevent_initial_call=True,
    )
    def set_active_page(
        n1: int,
        n2: int,
        n3: int,
        n4: int,
        n5: int,
        n6: int,
        n7: int,
        n_img: int,
    ) -> str:
        _ = (n1, n2, n3, n4, n5, n6, n7, n_img)
        id_to_page = {
            "menu-step1": "step1",
            "menu-step2": "step2",
            "menu-step3": "step3",
            "menu-step4": "step4",
            "menu-step5": "step5",
            "menu-step6": "step6",
            "menu-step7": "step7",
            "menu-samples": "samples",
        }
        return id_to_page.get(dash_ctx.triggered_id, "step1")

    @app.callback(
        Output("menu-header", "children"),
        Output("page-content", "children"),
        Input("active-page", "data"),
        Input("filter-split", "value"),
        Input("filter-class", "value"),
        Input("sample-limit", "value"),
    )
    def render_content(active_page: str, split_filter: str, class_filter: str, sample_limit: int):
        selected = active_page or "step1"

        if selected == "step1":
            title = STEP_LABELS["step1"]
            comp = step1_split.render(ctx, split_filter, class_filter)
        elif selected in static_pages:
            title = STEP_LABELS[selected]
            comp = static_pages[selected]
        else:
            title = "Tab ảnh mẫu từ train/val/test"
            comp = step_samples.render(split_filter, class_filter, int(sample_limit))

        header = html.Div(
            children=[
                html.H1("Phát hiện Viêm phổi từ X-quang - EDA Dashboard", className="main-title"),
                html.P(f"Menu hiện tại: {title}", className="main-subtitle"),
            ]
        )
        return header, comp

    return app


def run() -> None:
    app = create_app()
    port = int(os.getenv("EDA_DASHBOARD_PORT", "8080"))
    debug = os.getenv("EDA_DASHBOARD_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
    app.run(debug=debug, host="0.0.0.0", port=port)


if __name__ == "__main__":
    run()
