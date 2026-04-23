"""
=============================================================================
  Glasgow Weather Data — Climate Change Temperature Visualizations
  Dataset : Glasgow Weather Data 2015-2019 (Kaggle)
  Tools   : Pandas, Matplotlib, Seaborn
=============================================================================

  Run once to install dependencies (if not already installed):
      pip install pandas matplotlib seaborn numpy

  Then run:
      python climate_visualization.py

  Visualisations produced
  -----------------------
  Fig 1  Daily Temperature Range           -- Line chart + fill-between
  Fig 2  Monthly Average Temperature Trend -- Line chart with min/max band
  Fig 3  Monthly x Year Heatmap            -- Seasonal pattern at a glance
  Fig 4  Avg Temp vs Humidity              -- Scatter, coloured by year
  Fig 5  Avg Temp vs Wind Speed            -- Scatter, coloured by season
  Fig 6  Monthly Temperature Distribution  -- Box plot (all years)
  Fig 7  Year-on-Year Monthly Overlay      -- Line chart per year
"""

# ---- Auto-install missing packages -----------------------------------------
import subprocess
import sys

REQUIRED = ["pandas", "matplotlib", "seaborn", "numpy"]

for pkg in REQUIRED:
    try:
        __import__(pkg)
    except ImportError:
        print(f"  Installing missing package: {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# ---- Imports ----------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np

# ---- Constants --------------------------------------------------------------
DATA_PATH  = "clean_weather_data.csv"   # update path if the CSV is elsewhere
OUTPUT_DIR = "."                         # folder where PNGs will be saved

COLD   = '#4A90D9'
HOT    = '#E05C5C'
AVG    = '#2ECC71'
ACCENT = '#F39C12'
BG     = '#F8F9FA'

MONTH_LABELS = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec']

YEAR_COLORS  = {2015:'#1f77b4', 2016:'#ff7f0e', 2017:'#2ca02c',
                2018:'#d62728', 2019:'#9467bd'}

SEASON_MAP = {12:'Winter', 1:'Winter',  2:'Winter',
               3:'Spring',  4:'Spring',  5:'Spring',
               6:'Summer',  7:'Summer',  8:'Summer',
               9:'Autumn', 10:'Autumn', 11:'Autumn'}

SEASON_COLORS = {
    'Winter': COLD,
    'Spring': AVG,
    'Summer': HOT,
    'Autumn': ACCENT
}

sns.set_theme(style='whitegrid', palette='deep')


# ---- 1. Load & Prepare Data -------------------------------------------------
def load_data(path):
    """Read CSV and engineer helper columns."""
    df = pd.read_csv(path)
    df['day']     = pd.to_datetime(df['day'])
    df['tempAvg'] = (df['tempMin'] + df['tempMax']) / 2
    df['month']   = df['day'].dt.month
    df['year']    = df['day'].dt.year
    df['season']  = df['month'].map(SEASON_MAP)
    return df


def monthly_aggregate(df):
    """Resample daily data to monthly mean temperatures."""
    return (
        df.resample('ME', on='day')
          .agg(tempMin=('tempMin', 'mean'),
               tempMax=('tempMax', 'mean'),
               tempAvg=('tempAvg', 'mean'))
          .reset_index()
    )


# ---- 2. Plot Helpers --------------------------------------------------------
def styled_fig(figsize=(14, 5)):
    """Return a (fig, ax) pair with the project background colour."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    return fig, ax


def save(fig, name):
    """Save figure as PNG and close it."""
    path = f"{OUTPUT_DIR}/{name}"
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved -> {path}")


# ---- 3. Plot Functions ------------------------------------------------------

def plot_daily_range(df):
    """Fig 1 -- Daily temperature range as a filled line chart."""
    fig, ax = styled_fig(figsize=(16, 6))

    ax.fill_between(df['day'], df['tempMin'], df['tempMax'],
                    alpha=0.15, color=ACCENT, label='Daily Range')
    ax.plot(df['day'], df['tempMax'], color=HOT,  lw=1.2, alpha=0.8, label='Max Temp')
    ax.plot(df['day'], df['tempMin'], color=COLD, lw=1.2, alpha=0.8, label='Min Temp')
    ax.plot(df['day'], df['tempAvg'], color=AVG,  lw=1.8, alpha=0.95, label='Avg Temp')

    ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.5, label='0 C')
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))

    ax.set_title('Glasgow Daily Temperature Range (2015-2019)',
                 fontsize=16, fontweight='bold', pad=14)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (C)', fontsize=12)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    save(fig, 'fig1_daily_temp_range.png')


def plot_monthly_trend(monthly):
    """Fig 2 -- Monthly average temperature trend with min/max band."""
    fig, ax = styled_fig()

    ax.fill_between(monthly['day'], monthly['tempMin'], monthly['tempMax'],
                    alpha=0.20, color=ACCENT, label='Min-Max Band')
    ax.plot(monthly['day'], monthly['tempMax'], color=HOT,  lw=2,
            marker='o', markersize=3, label='Monthly Max')
    ax.plot(monthly['day'], monthly['tempMin'], color=COLD, lw=2,
            marker='o', markersize=3, label='Monthly Min')
    ax.plot(monthly['day'], monthly['tempAvg'], color=AVG,  lw=2.5,
            marker='D', markersize=4, label='Monthly Avg', zorder=5)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_title('Monthly Average Temperature Trend -- Glasgow',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Temperature (C)', fontsize=12)
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    save(fig, 'fig2_monthly_trend.png')


def plot_heatmap(df):
    """Fig 3 -- Month x Year heatmap of average temperatures."""
    pivot = df.pivot_table(values='tempAvg', index='month',
                           columns='year', aggfunc='mean')

    fig, ax = styled_fig(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlBu_r',
                linewidths=0.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Avg Temp (C)'})
    ax.set_yticklabels(MONTH_LABELS, rotation=0, fontsize=11)
    ax.set_xticklabels(pivot.columns, rotation=0, fontsize=11)

    ax.set_title('Monthly Average Temperature Heatmap (C)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Month', fontsize=12)

    plt.tight_layout()
    save(fig, 'fig3_heatmap.png')


def plot_scatter_humidity(df):
    """Fig 4 -- Scatter: avg temp vs humidity, coloured by year."""
    fig, ax = styled_fig(figsize=(10, 6))

    scatter = ax.scatter(df['humidity'], df['tempAvg'],
                         c=df['year'], cmap='viridis',
                         alpha=0.4, s=18, edgecolors='none')

    z = np.polyfit(df['humidity'], df['tempAvg'], 1)
    x_line = np.linspace(df['humidity'].min(), df['humidity'].max(), 200)
    ax.plot(x_line, np.poly1d(z)(x_line),
            color='red', lw=2, ls='--', label='Trend line', zorder=5)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Year', fontsize=11)
    cbar.set_ticks([2015, 2016, 2017, 2018, 2019])

    ax.set_title('Average Temperature vs Humidity (coloured by Year)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Humidity', fontsize=12)
    ax.set_ylabel('Average Temperature (C)', fontsize=12)
    ax.legend(framealpha=0.9)

    plt.tight_layout()
    save(fig, 'fig4_scatter_humidity.png')


def plot_scatter_wind(df):
    """Fig 5 -- Scatter: avg temp vs wind speed, coloured by season."""
    fig, ax = styled_fig(figsize=(10, 6))

    for season, grp in df.groupby('season'):
        ax.scatter(grp['windSpeed'], grp['tempAvg'],
                   color=SEASON_COLORS[season], alpha=0.4,
                   s=18, label=season, edgecolors='none')

    ax.set_title('Average Temperature vs Wind Speed (by Season)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Wind Speed (km/h)', fontsize=12)
    ax.set_ylabel('Average Temperature (C)', fontsize=12)
    ax.legend(title='Season', framealpha=0.9)

    plt.tight_layout()
    save(fig, 'fig5_scatter_wind.png')


def plot_boxplot_monthly(df):
    """Fig 6 -- Box plot: temperature distribution per calendar month."""
    fig, ax = styled_fig(figsize=(14, 6))

    sns.boxplot(data=df, x='month', y='tempAvg',
                hue='month', palette='coolwarm', legend=False,
                ax=ax, linewidth=1.2, fliersize=2)
    ax.set_xticks(range(12))
    ax.set_xticklabels(MONTH_LABELS, fontsize=11)

    ax.set_title('Monthly Temperature Distribution (2015-2019)',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Temperature (C)', fontsize=12)

    plt.tight_layout()
    save(fig, 'fig6_boxplot_month.png')


def plot_yoy_overlay(df):
    """Fig 7 -- Year-on-year monthly temperature overlay."""
    fig, ax = styled_fig(figsize=(12, 6))

    for yr, grp in df.groupby('year'):
        grp_m = grp.groupby('month')['tempAvg'].mean()
        ax.plot(grp_m.index, grp_m.values,
                marker='o', lw=2.2,
                color=YEAR_COLORS[yr], label=str(yr))

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=11)
    ax.set_title('Year-on-Year Monthly Temperature Comparison',
                 fontsize=15, fontweight='bold', pad=12)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Average Temperature (C)', fontsize=12)
    ax.legend(title='Year', framealpha=0.9)

    plt.tight_layout()
    save(fig, 'fig7_yoy_overlay.png')


# ---- 4. Main ----------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Glasgow Climate Visualization -- starting")
    print("=" * 60)

    print("\n[1/2] Loading data ...")
    df      = load_data(DATA_PATH)
    monthly = monthly_aggregate(df)

    print(f"  Records  : {len(df):,}")
    print(f"  Period   : {df['day'].min().date()} -> {df['day'].max().date()}")
    print(f"  Columns  : {', '.join(df.columns)}")

    print("\n[2/2] Generating charts ...")
    plot_daily_range(df)
    plot_monthly_trend(monthly)
    plot_heatmap(df)
    plot_scatter_humidity(df)
    plot_scatter_wind(df)
    plot_boxplot_monthly(df)
    plot_yoy_overlay(df)

    print("\nDone -- 7 charts saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
