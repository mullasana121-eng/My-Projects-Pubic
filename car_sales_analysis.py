"""
============================================================
  Car Sales Data Analysis
  Tools: Pandas, Matplotlib, Plotly
  Dataset: Car_sales.csv (Kaggle - gagandeep16/car-sales)
============================================================

IMPLEMENTATION PROCEDURE
─────────────────────────
Step 1 : Install required libraries
         pip install pandas matplotlib plotly

Step 2 : Place 'Car_sales.csv' in the same folder as this script
         (Download from https://www.kaggle.com/datasets/gagandeep16/car-sales)

Step 3 : Run the script
         python car_sales_analysis.py

Step 4 : Four chart windows will open (Matplotlib), then an
         interactive browser tab will open (Plotly).
         HTML file 'car_sales_plotly_charts.html' is also saved locally.
"""

# ──────────────────────────────────────────────────────────
# 0.  Imports
# ──────────────────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────
# 1.  Load & Clean Data
# ──────────────────────────────────────────────────────────
df = pd.read_csv("Car_sales.csv")

# Drop rows where Sales_in_thousands is missing
df_clean = df.dropna(subset=["Sales_in_thousands"]).copy()

# Full model label: "Ford F-Series"
df_clean["Full_Model"] = df_clean["Manufacturer"] + " " + df_clean["Model"]

print(f"Dataset loaded: {len(df_clean)} records across {df_clean['Manufacturer'].nunique()} manufacturers\n")

# ──────────────────────────────────────────────────────────
# 2.  Aggregate Data
# ──────────────────────────────────────────────────────────
# Top 15 individual car models by sales
top_models = (
    df_clean.nlargest(15, "Sales_in_thousands")[["Full_Model", "Sales_in_thousands"]]
    .reset_index(drop=True)
)

# Sales by manufacturer (all)
mfr_sales = (
    df_clean.groupby("Manufacturer")["Sales_in_thousands"]
    .sum()
    .sort_values(ascending=False)
)

# Top 10 manufacturers for pie chart readability
top10_mfr = mfr_sales.head(10)
others_sum = mfr_sales.iloc[10:].sum()
pie_labels = list(top10_mfr.index) + ["Others"]
pie_values = list(top10_mfr.values) + [others_sum]

# Sales by vehicle type
type_sales = df_clean.groupby("Vehicle_type")["Sales_in_thousands"].sum().sort_values(ascending=False)

# ──────────────────────────────────────────────────────────
# 3.  Matplotlib Charts
# ──────────────────────────────────────────────────────────
PALETTE = [
    "#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED",
    "#0891B2", "#BE185D", "#65A30D", "#EA580C", "#0D9488",
    "#9333EA", "#C026D3", "#B45309", "#475569", "#6B7280",
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Chart 1 : Horizontal Bar — Top 15 Car Models ──────────
fig1, ax1 = plt.subplots(figsize=(12, 7))
colors1 = PALETTE[: len(top_models)]

bars = ax1.barh(
    top_models["Full_Model"][::-1],        # reverse so #1 is on top
    top_models["Sales_in_thousands"][::-1],
    color=colors1[::-1],
    edgecolor="white",
    linewidth=0.6,
)

for bar in bars:
    width = bar.get_width()
    ax1.text(
        width + 3, bar.get_y() + bar.get_height() / 2,
        f"{width:,.1f}k", va="center", ha="left", fontsize=9, color="#374151",
    )

ax1.set_xlabel("Sales (thousands)", fontsize=11)
ax1.set_title("Top 15 Best-Selling Car Models", fontsize=14, fontweight="bold", pad=14)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}k"))
ax1.set_xlim(0, top_models["Sales_in_thousands"].max() * 1.18)
plt.tight_layout()
plt.savefig("chart1_top15_models.png", dpi=150)
print("Saved: chart1_top15_models.png")

# ── Chart 2 : Vertical Bar — Top 10 Manufacturers ─────────
fig2, ax2 = plt.subplots(figsize=(11, 6))
x_pos = range(len(top10_mfr))

ax2.bar(x_pos, top10_mfr.values, color=PALETTE[:10], edgecolor="white", linewidth=0.6)

for i, (val, label) in enumerate(zip(top10_mfr.values, top10_mfr.index)):
    ax2.text(i, val + 10, f"{val:,.0f}k", ha="center", fontsize=8.5, color="#374151")

ax2.set_xticks(list(x_pos))
ax2.set_xticklabels(top10_mfr.index, rotation=30, ha="right", fontsize=10)
ax2.set_ylabel("Total Sales (thousands)", fontsize=11)
ax2.set_title("Total Sales by Manufacturer — Top 10", fontsize=14, fontweight="bold", pad=14)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}k"))
plt.tight_layout()
plt.savefig("chart2_manufacturer_bar.png", dpi=150)
print("Saved: chart2_manufacturer_bar.png")

# ── Chart 3 : Pie Chart — Manufacturer Market Share ───────
fig3, ax3 = plt.subplots(figsize=(10, 8))
wedge_props = dict(width=0.6, edgecolor="white", linewidth=1.5)   # donut style

wedges, texts, autotexts = ax3.pie(
    pie_values,
    labels=None,
    autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
    startangle=140,
    colors=PALETTE[: len(pie_labels)],
    wedgeprops=wedge_props,
    pctdistance=0.75,
)

for at in autotexts:
    at.set_fontsize(8)
    at.set_color("white")
    at.set_fontweight("bold")

ax3.legend(
    wedges, pie_labels,
    title="Manufacturer", title_fontsize=10,
    loc="center left", bbox_to_anchor=(1.02, 0.5),
    fontsize=9, frameon=False,
)
ax3.set_title("Market Share by Manufacturer", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("chart3_manufacturer_pie.png", dpi=150)
print("Saved: chart3_manufacturer_pie.png")

# ── Chart 4 : Pie Chart — Vehicle Type Distribution ───────
fig4, ax4 = plt.subplots(figsize=(8, 6))

wedges2, texts2, autotexts2 = ax4.pie(
    type_sales.values,
    labels=type_sales.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=PALETTE[: len(type_sales)],
    wedgeprops=dict(edgecolor="white", linewidth=1.5),
    pctdistance=0.6,
)

for at in autotexts2:
    at.set_fontsize(10)
    at.set_color("white")
    at.set_fontweight("bold")

ax4.set_title("Sales Distribution by Vehicle Type", fontsize=14, fontweight="bold", pad=14)
plt.tight_layout()
plt.savefig("chart4_vehicle_type_pie.png", dpi=150)
print("Saved: chart4_vehicle_type_pie.png")

plt.show()

# ──────────────────────────────────────────────────────────
# 4.  Plotly Interactive Charts  (opens in browser)
# ──────────────────────────────────────────────────────────
fig_plotly = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Top 15 Car Models by Sales",
        "Manufacturer Market Share",
        "Top 10 Manufacturers — Total Sales",
        "Vehicle Type Distribution",
    ),
    specs=[
        [{"type": "bar"}, {"type": "pie"}],
        [{"type": "bar"}, {"type": "pie"}],
    ],
    vertical_spacing=0.14,
    horizontal_spacing=0.10,
)

# Sub-plot 1: Top 15 models (horizontal bar)
fig_plotly.add_trace(
    go.Bar(
        y=top_models["Full_Model"],
        x=top_models["Sales_in_thousands"],
        orientation="h",
        marker_color=PALETTE[: len(top_models)],
        text=top_models["Sales_in_thousands"].round(1).astype(str) + "k",
        textposition="outside",
        name="Model Sales",
    ),
    row=1, col=1,
)

# Sub-plot 2: Manufacturer pie
fig_plotly.add_trace(
    go.Pie(
        labels=pie_labels,
        values=pie_values,
        hole=0.4,
        marker_colors=PALETTE[: len(pie_labels)],
        textinfo="percent+label",
        textfont_size=10,
        name="Manufacturer Share",
    ),
    row=1, col=2,
)

# Sub-plot 3: Manufacturer bar
fig_plotly.add_trace(
    go.Bar(
        x=list(top10_mfr.index),
        y=list(top10_mfr.values),
        marker_color=PALETTE[:10],
        text=[f"{v:,.0f}k" for v in top10_mfr.values],
        textposition="outside",
        name="Manufacturer Sales",
    ),
    row=2, col=1,
)

# Sub-plot 4: Vehicle type pie
fig_plotly.add_trace(
    go.Pie(
        labels=list(type_sales.index),
        values=list(type_sales.values),
        marker_colors=PALETTE[: len(type_sales)],
        textinfo="percent+label",
        name="Vehicle Type",
    ),
    row=2, col=2,
)

fig_plotly.update_layout(
    title_text="🚗  Car Sales Dashboard",
    title_font_size=20,
    height=850,
    showlegend=False,
    template="plotly_white",
    font=dict(family="Arial", size=11),
)

fig_plotly.write_html("car_sales_plotly_charts.html")
print("\nSaved interactive dashboard: car_sales_plotly_charts.html")
fig_plotly.show()

# ──────────────────────────────────────────────────────────
# 5.  Summary Statistics
# ──────────────────────────────────────────────────────────
print("\n" + "="*55)
print("  SUMMARY INSIGHTS")
print("="*55)
top1 = top_models.iloc[0]
print(f"  Best-selling model  : {top1['Full_Model']}  ({top1['Sales_in_thousands']:,.1f}k units)")
print(f"  Top manufacturer    : {mfr_sales.index[0]}  ({mfr_sales.iloc[0]:,.1f}k total)")
print(f"  Dominant type       : {type_sales.index[0]}  ({type_sales.iloc[0] / type_sales.sum() * 100:.1f}% of sales)")
print(f"  Total models tracked: {len(df_clean)}")
print(f"  Total units sold    : {df_clean['Sales_in_thousands'].sum():,.1f}k")
print("="*55)
