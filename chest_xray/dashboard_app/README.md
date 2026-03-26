# Dashboard App (7-step EDA)

## Run

```bash
conda activate ML
python chest_xray/eda_dashboard.py
```

Default port: `8088`.
Override:

```bash
EDA_DASHBOARD_PORT=8090 python chest_xray/eda_dashboard.py
```

## Structure

- `app.py`: main Dash app and menu navigation.
- `data_loader.py`: load metadata and image previews.
- `components/step1_split.py` ... `components/step7_evaluation.py`: 7 pipeline steps.
- `components/step_samples.py`: sample images tab.
- `assets/style.css`: dashboard styling.
