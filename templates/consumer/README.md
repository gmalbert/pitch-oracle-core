# Pitch Oracle League Consumer

This is the minimal scaffold for a league deployment backed by
`pitch-oracle-core`. Choose a registered league in `config.py`, add league-owned
data/assets where needed, and keep `CORE_REF` synchronized with the immutable tag
in `requirements.txt`.

Local verification:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
streamlit run predictions.py
```

Generated `data_files/`, `models/`, and `precomputed/` artifacts are produced by
the shared workflow and must be committed together with a strict cache manifest.
