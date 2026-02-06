# Plotly Renderer Configuration for Jupyter Notebooks
# Run this cell BEFORE the Interactive ROC Curve section to fix nbformat errors

import plotly.io as pio

# Set the default renderer for Jupyter notebooks
pio.renderers.default = "notebook"

print("✅ Plotly renderer configured successfully for Jupyter notebooks!")
print(f"   Default renderer: {pio.renderers.default}")
