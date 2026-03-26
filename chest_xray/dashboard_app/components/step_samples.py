from __future__ import annotations

from dash import html

from ..data_loader import list_sample_images


def render(split_filter: str, class_filter: str, sample_limit: int) -> html.Div:
    images = list_sample_images(split_filter, class_filter, sample_limit)
    if not images:
        return html.Div("Không tìm thấy ảnh mẫu theo filter hiện tại.", className="step-desc")

    cards = []
    for item in images:
        cards.append(
            html.Div(
                className="sample-card",
                children=[
                    html.Img(src=item["uri"], className="sample-img"),
                    html.Div(f"{item['split']} | {item['class']}", className="sample-meta"),
                    html.Div(item["name"], className="sample-name"),
                ],
            )
        )

    return html.Div(
        children=[
            html.H3("Tab ảnh mẫu NORMAL/PNEUMONIA", className="step-title"),
            html.P(
                "Ảnh được lấy từ các thư mục train/val/test theo filter đang chọn để kiểm tra trực quan chất lượng dữ liệu và độ đa dạng mẫu.",
                className="step-desc",
            ),
            html.Div(
                className="formula-box",
                children=[
                    html.Strong("Mục đích tab ảnh mẫu:"),
                    html.Span(" hỗ trợ rà soát nhanh vấn đề nhãn sai, ảnh mờ, ảnh trùng, và sự khác biệt phân phối giữa các split."),
                ],
            ),
            html.Div(className="sample-grid", children=cards),
        ]
    )
