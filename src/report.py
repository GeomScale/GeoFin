import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict

# =========================
# CONFIG
# =========================

class ReportConfig:
    COLOR_POSITIVE = '#2ecc71'
    COLOR_NEGATIVE = '#e74c3c'
    COLOR_NEUTRAL = '#3498db'

    FIGSIZE_SINGLE = (12, 6)
    FIGSIZE_MULTI = (16, 10)
    FIGSIZE_HEATMAP = (10, 6)

    FONT_FAMILY = 'sans-serif'
    FONT_SIZE = 10
    TITLE_SIZE = 14

    GRID_ALPHA = 0.3
    GRID_STYLE = '--'


def setup_style(config: ReportConfig):
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({
        'font.family': config.FONT_FAMILY,
        'font.size': config.FONT_SIZE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.facecolor': '#f5f5f5',
    })


# =========================
# METRICS
# =========================

class MetricsComputer:

    def __init__(self, risk_free_rate: float = 0.02):
        self.rf = risk_free_rate

    def _to_series(self, data) -> pd.Series:

        # Case 1: already a Series
        if isinstance(data, pd.Series):
            return data

        # Case 2: object has .returns (bt, vectorbt, etc.)
        if hasattr(data, "returns"):
            return data.returns

        # Case 3: object has .prices → convert to returns
        if hasattr(data, "prices"):
            return data.prices.pct_change().dropna()

        # Case 4: list / numpy
        if isinstance(data, (list, np.ndarray)):
            return pd.Series(data)

        raise TypeError(f"Unsupported type: {type(data)}")

    def _validate_returns(self, data) -> pd.Series:
        returns = self._to_series(data)

        returns = returns.dropna().astype(float)

        if returns.empty:
            raise ValueError("Empty returns")

        return returns

    def compute_metrics(self, sim: Dict) -> pd.DataFrame:
        metrics = {}

        for name, sim_obj in sim.items():
            try:
                returns = self._validate_returns(sim_obj)
                metrics[name] = self._compute(returns)

            except Exception as e:
                print(f"[WARN] {name}: {e}")

        if not metrics:
            raise ValueError("No valid strategies")

        df = pd.DataFrame(metrics).T
        return df.sort_values("Sharpe", ascending=False)

    def _compute(self, r: pd.Series) -> Dict:
        total_return = (1 + r).prod() - 1

        vol = r.std() * np.sqrt(252)

        sharpe = (r.mean() * 252 - self.rf) / (vol + 1e-9)

        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()

        calmar = total_return / abs(max_dd) if max_dd != 0 else np.nan

        return {
            "Return (%)": total_return * 100,
            "Volatility (%)": vol * 100,
            "Sharpe": sharpe,
            "Calmar": calmar,
            "Max Drawdown (%)": max_dd * 100,
            "Win Rate (%)": (r > 0).mean() * 100,
            "VaR (95%)": r.quantile(0.05) * 100,
        }


# =========================
# VISUALIZATION
# =========================

class BacktestVisualizer:

    def __init__(self, config: ReportConfig):
        self.config = config
        setup_style(config)

    def plot_heatmap(self, df: pd.DataFrame):

        denom = (df.max() - df.min()).replace(0, np.nan)
        norm = (df - df.min()) / denom

        plt.figure(figsize=self.config.FIGSIZE_HEATMAP)

        sns.heatmap(
            norm,
            annot=df.round(2),
            fmt='',
            cmap='RdYlGn',
            linewidths=0.5
        )

        plt.title("Metrics Heatmap")
        plt.tight_layout()
        plt.show()

    def plot_cum_returns(self, sim: Dict):

        plt.figure(figsize=self.config.FIGSIZE_SINGLE)

        mc = MetricsComputer()

        for name, sim_obj in sim.items():
            try:
                r = mc._validate_returns(sim_obj)
                cum = (1 + r).cumprod()
                plt.plot(cum.index, cum.values, label=name)
            except Exception as e:
                print(f"[WARN] {name}: {e}")

        plt.title("Cumulative Returns")
        plt.legend()
        plt.show()
        

    def plot_drawdown(self, sim: Dict):

        plt.figure(figsize=self.config.FIGSIZE_SINGLE)

        mc = MetricsComputer()

        for name, sim_obj in sim.items():
            try:
                r = mc._validate_returns(sim_obj)

                cum = (1 + r).cumprod()
                peak = cum.cummax()
                dd = (cum - peak) / peak

                plt.plot(dd.index, dd.values, label=name)
            except Exception as e:
                print(f"[WARN] {name}: {e}")

        plt.title("Drawdown")
        plt.legend()
        plt.show()
        
        
    def plot_bar(self, df: pd.DataFrame):
        metrics = ["Sharpe", "Calmar", "Max Drawdown (%)", "VaR (95%)"]

        df_plot = df[metrics]

        df_plot.plot(
            kind='bar',
            subplots=True,
            layout=(2, 2),
            figsize=(12, 6),
            legend=False,
            sharex=True
        )

        plt.suptitle("Performance Metrics (Bar)")
        plt.tight_layout()
        plt.show()

# =========================
# REPORT
# =========================

class BacktestReporter:

    def __init__(self):
        self.config = ReportConfig()
        self.comp = MetricsComputer()
        self.vis = BacktestVisualizer(self.config)

    def run(self, sim: Dict[str, pd.Series]) -> pd.DataFrame:

        df = self.comp.compute_metrics(sim)

        print("\n===== METRICS =====")
        print(df.round(3))

        self.vis.plot_cum_returns(sim)
        self.vis.plot_drawdown(sim)
        self.vis.plot_heatmap(df)
        self.vis.plot_bar(df)

        return df


# =========================
# QUICK USE
# =========================

def quick_report(sim: Dict[str, pd.Series]):
    return BacktestReporter().run(sim)