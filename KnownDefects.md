### Known bugs:
- In `ml_interpretations.py`
1. Posx and posy should be finite values. Text and fig scaling issues.
2. Shap - matplotlib = True is not yet supported for force plots with multiple samples! Example: Pick [Personal ID 53716]
3. Segmentation fault. Sometimes it crashes.

- In `des_statistics.py`
1. Basic Enrollment Stats - if the date range changes, and if within those observations the number of possible genders changes, the colors indicating gender in the legend may change. 
2. Exit Outcome Facet Chart.  If the date range selected by the user does not contain at least one example of each exit outcome, the chart will fail to render.
3. Exit Outcome Facet Chart.  if the user zooms the 3 charts will become "unmoored" - the x axis moves.

- In `main.py`
1. Needs debugging. Not functional with Streamlit app