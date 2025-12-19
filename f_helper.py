import warnings

# Third-party imports (PEP8: stdlib first, third-party next)
from adjustText import adjust_text
import numpy as np
from numpy import log
import pandas as pd
import pylab as plt
import seaborn as sns
from scipy.stats import kendalltau
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

# Silence noisy RuntimeWarnings from numpy/pandas when taking logs of non-positive
# values. These warnings occur in many places where log is applied to series that
# may contain zeros/NaNs; we intentionally ignore the `invalid value encountered
# in log` RuntimeWarning to reduce noise.
np.seterr(invalid='ignore')
warnings.filterwarnings("ignore", message="invalid value encountered in log", category=RuntimeWarning)

source_to_label = {"BEA_income": "BoP Income + foreign taxes",
                  "BEA_return": "BEA: Profit-type return",
                   "BEA_return_cg": "BEA: Profit-type return + capital gains",
                  "CFC": "CFC Form 5471: E&P - dividends",
                  "CBCR": "CBCR: Profits"}

def plot_pi_emp_detail(df, source, iso3_to_name, dev_haven, island_states):
    """Plot profit-per-employee vs ETR for a given `source`.

    Parameters
    - df: full dataframe containing columns used below
    - source: data source string (e.g., 'CBCR')
    - iso3_to_name: callable mapping iso3 -> human-readable name
    - dev_haven, island_states: lists of iso3 codes used for highlighting

    Returns the last plotted merged dataframe (for inspection).
    """

    # Work on a copy to avoid mutating the caller's dataframe
    data_source = df.loc[df["source"] == source].copy()

    # Keep ETR on a percentage scale (multiply by 100)
    data_source["etr"] = data_source["etr"] * 100

    last_merged = None

    for year, group in data_source.groupby("year"):
        # Select the years of interest depending on the source
        if source != "CFC" and year not in [2017, 2018, 2019]:
            continue
        if source == "CFC" and year != 2016:
            continue

        # Choose positive_groups for CBCR/CFC where applicable, otherwise use all_groups
        pos_mask = group["type"] == "positive_groups" if source in ["CBCR", "CFC"] else group["type"] == "all_groups"
        x_df = group.loc[pos_mask]
        y_df = group.loc[group["type"] == "all_groups"]

        # Merge to ensure we have the 'all_groups' values available with suffix _y
        merged = pd.merge(x_df, y_df, on=["iso3", "year"], suffixes=("", "_y"), how="right")

        # Fill key missing values from the 'all_groups' columns where needed
        for col in ["etr", "pi_emp", "pi", "txc"]:
            merged[col] = merged[col].fillna(merged.get(col + "_y"))

        # Remove aggregated/other labels and isolate Rest and actual countries
        merged = merged.loc[~merged["iso3"].str.contains("Other ")]
        total_foreign_profits = merged.loc[merged["iso3"] == "Rest", "pi_y"].sum()
        merged = merged.loc[merged["iso3"] != "Rest"]

        texts_island = []
        texts_dev = []

        plt.figure(figsize=(6, 4))
        ax = plt.subplot(1, 1, 1)

        # Scatter: marker size roughly proportional to sqrt(pi)
        plt.scatter(merged["etr"], merged["pi_emp"], s=2 * np.sqrt(merged["pi"]) / 10000, color="gray")

        # Highlight developer havens
        th = 1E5
        dev_subset = merged.loc[merged["iso3"].isin(dev_haven) & (merged["pi"] > 0)]
        for _, row in dev_subset.iterrows():
            if row["pi"] > 10E9:
                texts_dev.append(plt.text(row["etr"], row["pi_emp"], iso3_to_name(row["iso3"]), color="gray", fontsize=10))

        plt.hlines([th, 2E6], 5, 15, color="orange")
        plt.vlines([5, 15], th, 2E6, color="orange")

        if len(dev_subset):
            prof_perc = dev_subset["pi_y"].sum() / (total_foreign_profits if total_foreign_profits else 1)
            prof_per_employee = dev_subset["pi"].sum() / (dev_subset["emp"].sum() if dev_subset["emp"].sum() else 1)
            etr_dev = dev_subset["txc"].sum() / (dev_subset["pi"].sum() if dev_subset["pi"].sum() else 1)
            plt.annotate(
                f"${prof_per_employee:2,.0f} per employee\n{prof_perc:2.1%} of total foreign profits\nEffective tax rate: {etr_dev:2.1%}",
                (16, 1E6), color="orange", fontsize=10, va="center", ha="left", xytext=(20, 1E6),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="orange"),
            )
            plt.scatter(dev_subset["etr"], dev_subset["pi_emp"], s=2 * np.sqrt(dev_subset["pi"]) / 10000, color="orange")

        # Highlight island states
        isl_subset = merged.loc[merged["iso3"].isin(island_states) & (merged["pi"] > 0)]
        for _, row in isl_subset.iterrows():
            if row["pi"] > 10E9:
                name = iso3_to_name(row["iso3"])
                if name == "BEA_Other_Western":
                    name = "Other Western"
                texts_island.append(plt.text(row["etr"], row["pi_emp"], name, color="gray", fontsize=10))

        plt.hlines([th, 4E8], -0.5, 4, color="tomato")
        plt.vlines([-0.5, 4], th, 4E8, color="tomato")

        if len(isl_subset):
            prof_perc_isl = isl_subset["pi_y"].sum() / (total_foreign_profits if total_foreign_profits else 1)
            prof_isl = isl_subset["pi"].sum() / (isl_subset["emp"].sum() if isl_subset["emp"].sum() else 1)
            etr_isl = isl_subset["txc"].sum() / (isl_subset["pi"].sum() if isl_subset["pi"].sum() else 1)
            plt.annotate(
                f"${prof_isl:2,.0f} per employee\n{prof_perc_isl:2.1%} of total foreign profits\nEffective tax rate: {etr_isl:2.1%}",
                (5, 1E8), color="tomato", fontsize=10, va="center", ha="left", xytext=(15, 1E8),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="tomato"),
            )
            plt.scatter(isl_subset["etr"], isl_subset["pi_emp"], s=2 * np.sqrt(isl_subset["pi"]) / 10000, color="tomato")

        # Annotate a couple of notable countries
        for label_iso, label_text, y_off in [("GBR", "UK", 0.7), ("USA", "USA", 0.6)]:
            country_row = merged.loc[merged["iso3"] == label_iso]
            if not country_row.empty:
                plt.scatter(country_row["etr"], country_row["pi_emp"], s=2 * np.sqrt(country_row["pi"]) / 10000,
                            color=("brown" if label_iso == "GBR" else "cornflowerblue"))
                plt.annotate(label_text, (country_row["etr"].values, country_row["pi_emp"].values * y_off),
                             va="top", ha="center", color=("brown" if label_iso == "GBR" else "cornflowerblue"), fontsize=10)

        plt.yscale("log")
        plt.xlim(-8, 50)
        plt.vlines(0, 2E3, 5E8, color="lightgray", zorder=0)
        ax.grid(axis='y')
        sns.despine(bottom=True, left=True)
        plt.ylim(2E3, 5E8)

        plt.ylabel("Profit per employee", fontsize=12)
        plt.xlabel("Effective tax rate (%)", fontsize=12)
        plt.tight_layout()

        # Use adjust_text to avoid overlapping labels
        if texts_island:
            adjust_text(texts_island, merged["etr"].values, merged["pi_emp"].values,
                        arrowprops=dict(arrowstyle='->', color='gray'), expand_points=(1., 0.7),
                        force_points=(0.9, 0.6), force_text=(0.7, 0.9), ha="right", autoalign="y", lim=1000)
        if texts_dev:
            adjust_text(texts_dev, dev_subset["etr"].values if len(dev_subset) else [],
                        dev_subset["pi_emp"].values if len(dev_subset) else [],
                        arrowprops=dict(arrowstyle='->', color='gray'), expand_points=(1.2, 1.2),
                        force_points=(0.4, 0.4), force_text=(0.6, 0.8), ha="left", autoalign="y", lim=100)

        last_merged = merged

    return last_merged


#Semi-elasticity
def run_regression(df,formula,bic=False,val=0.001,display_b=True,rate="etr", cou=""):
    """Run an OLS regression and return formatted effect strings and results.

    Parameters
    - df: pandas DataFrame containing the data
    - formula: patsy-style regression formula (string)
    - bic: include BIC display (bool)
    - val: small value used when computing semi-elasticities
    - display_b: whether to print/display short summary
    - rate: name of rate variable used in labels (e.g., 'etr' or 'metr')

    Returns
    - The function keeps the original behavior but documents inputs/outputs
        for clarity. When `display_b` is True it prints/display pieces and
        returns nothing; otherwise it returns computed effect strings and
        results objects (maintains backward compatibility with call sites).
    """

    mod_log = smf.ols(formula=formula,data=df)
    results_log = mod_log.fit()
    if bic:
        return results_log.bic,results_log.aic
    
    effect = "0"
    effect_cat = "0"
    deffect = "0"
    for p,i in results_log.params.items():
#         print(i,p)

        p = p.replace("haven[T.True]","haven")
    
        if ("log(1-" in p.replace(" ","")):
            if "T." in p:
                effect_cat += " + {}*log(1-{})*{}".format(i,rate,cou)
            else:
                effect_cat += " + {}*log(1-{})".format(i,rate)
                effect += " + {}*log(1-{})".format(i,rate)
                deffect += "- {}/(1-{})".format(i,rate)     
        elif ("log(0" in p) or ("log(1" in p) or ("e-0" in p): #handle both 0.0001 and 1E-4
            if "T." in p:
                effect_cat += " + {}*log({}+{})*{}".format(i,val,rate,cou)
            elif (f"- {rate}" in p) or (f"-{rate}" in p): 
                effect_cat += " + {}*log({}+{})".format(i,val,rate)
                effect += " + {}*log({}+{})".format(i,val,rate)
                deffect += "- {}/({}+{})".format(i,val,rate)
            else:
                effect_cat += " + {}*log({}+{})".format(i,val,rate)
                effect += " + {}*log({}+{})".format(i,val,rate)
                deffect += "+ {}/({}+{})".format(i,val,rate)
 
        elif (f"{rate} ** 2" in p) or (f"- {rate}) ** 2" in p):
            if "T." in p:
                effect_cat += " + {}*{}**2*{}".format(i,rate,cou)
            elif (f"- {rate}" in p) or (f"-{rate}" in p):
                effect_cat += " + {}*(1-{})**2".format(i,rate)
                effect += " + {}*(1-{})**2".format(i,rate)
                deffect += " - {}*2*{}".format(i,rate)
            else:
                effect_cat += " + {}*{}**2".format(i,rate)
                effect += " + {}*{}**2".format(i,rate)
                deffect += " + {}*2*{}".format(i,rate)

        elif "** (-3)" in p:
            if "T." in p:
                effect_cat += " + {}*({}+{})**(-3)*{}".format(i,val,rate,cou)
            else:
                effect_cat += " + {}*({}+{})**(-3)".format(i,val,rate)
                effect += " + {}*({}+{})**(-3)".format(i,val,rate)
                deffect += " +{}*-3*({}+{})**(-4)".format(i,val,rate) 
        elif "** (-2)" in p:
            if "T." in p:
                effect_cat += " + {}*({}+{})**(-2)*{}".format(i,val,rate,cou)
            else:
                effect_cat += " + {}*({}+{})**(-2)".format(i,val,rate)
                effect += " + {}*({}+{})**(-2)".format(i,val,rate)
                deffect += "+{}*-2*({}+{})**(-3)".format(i,val,rate)
        elif "** (-1)" in p:
            if "T." in p:
                effect_cat += " + {}*({}+{})**(-1)*{}".format(i,val,rate,cou)
            else:
                effect_cat += " + {}*({}+{})**(-1)".format(i,val,rate)
                effect += " + {}*({}+{})**(-1)".format(i,val,rate)
                deffect += "+{}*-1*({}+{})**(-2)".format(i,val,rate)
        elif "coth" in p:
            if "T." in p:
                effect_cat += " + {}*coth({}+{})*{}".format(i,val,rate,cou)
            else:
                effect_cat += " + {}*coth({}+{})".format(i,val,rate)
                effect += " + {}*coth({}+{})".format(i,val,rate)
                deffect += " - (csch({}+{}))**2".format(val, rate)
        elif (f"- {rate}" in p) or (f"-{rate}" in p):
            if "T." in p:
                effect_cat += " + {}*(1-{})*{}".format(i,rate,cou)
            else:
                effect_cat += " + {}*(1-{})".format(i,rate)
                effect += " + {}*(1-{})".format(i,rate)
                deffect += " - {}".format(i) 
        elif rate in p:
            if "haven" in p:
                effect_cat += " + {}*{}*haven".format(i,rate)
                effect += " + {}*{}*haven".format(i,rate)
                deffect += " + {}*haven".format(i)                 
            else:
                effect_cat += " + {}*{}".format(i,rate)
                effect += " + {}*{}".format(i,rate)
                deffect += " + {}".format(i) 
        elif "haven" in p:
            effect_cat += " + {}*haven".format(i)
            effect += " + {}*haven".format(i)
            deffect += " + 0"

            
            
    if display_b:
        print(effect)
        print(results_log.summary())
    return effect,deffect,effect_cat,results_log


    

def plot_effect(formulas, axes=None, label="", color="gray", div=None, ref_etr=0.25, style='-', rate="etr", model="log"):
    """Plot the (non-linear) effect curve implied by a formula string.

    Arguments
    - formulas: sequence where formulas[0] is the effect expression (string) and
                formulas[1] is the marginal / derivative expression (string).
      These strings are evaluated with variables {rate, log, haven} present.
    - axes: optional iterable of matplotlib axes. If provided and length >= 4,
            uses axes[2], axes[3], axes[0], axes[1] to match previous layout.
    - div: two-element list-like [scale, offset] used to normalize the effect.
    - ref_etr: reference ETR used when comparing levels (baseline).
    - rate: name of the variable used inside formula strings (default 'etr' or 'metr').

    Notes:
    - This function uses Python's `eval` to evaluate formula strings. The
      evaluation context exposes `log` from numpy and optionally `haven`.
      Keep formulas under control to avoid security risks!
    """

    if div is None:
        div = [1, 0]

    sns.despine(bottom=True, left=True, right=True)
    plt.xlabel("$1 - \\tau $")
    plt.ylabel("Profits in country \n(compared with setting the tax rate\n of the country to 25%)")

    # build evaluation grid
    if rate == "metr":
        etr_grid = 1 - np.linspace(0.015, 0.5, 1000)
    else:
        etr_grid = np.linspace(0.015, 0.5, 1000)

    # helper to evaluate a formula string at a value `val` and optional haven flag
    def _eval_expr(expr, val, haven_flag=None):
        local_ctx = {rate: val, "log": log}
        if haven_flag is not None:
            local_ctx["haven"] = 1 if haven_flag else 0
        return eval(expr, {}, local_ctx)

    # Compute effect curve (level)
    if "haven" in formulas[0]:
        eff_haven = np.array([_eval_expr(formulas[0], v, haven_flag=1) for v in etr_grid]) + div[1]
        eff_haven = np.exp(eff_haven) / np.exp(_eval_expr(formulas[0], ref_etr, haven_flag=1)) / div[0]
        eff_non = np.array([_eval_expr(formulas[0], v, haven_flag=0) for v in etr_grid]) + div[1]
        eff_non = np.exp(eff_non) / np.exp(_eval_expr(formulas[0], ref_etr, haven_flag=0)) / div[0]
    else:
        eff = np.array([_eval_expr(formulas[0], v) for v in etr_grid])
        eff = np.exp(eff) / np.exp(_eval_expr(formulas[0], ref_etr)) / div[0]

    # Plot results (respect provided axes layout if available)
    if axes is not None and len(axes) >= 4:
        ax_marg, _, ax_eff, ax_eff_zoom = axes[0], axes[1], axes[2], axes[3]
    else:
        # create local axes layout
        fig, (ax_marg, ax_eff) = plt.subplots(1, 2, figsize=(10, 4))
        ax_eff_zoom = ax_eff

    if "haven" in formulas[0]:
        ax_eff.plot(etr_grid, eff_haven, "--", label=label + ": Havens", color=color)
        ax_eff.plot(etr_grid, eff_non, style, label=label + ": Non-havens", color=color)
        # zoomed plots (if axes provided, use slices)
        ax_eff_zoom.plot(etr_grid, eff_haven, "--", color=color)
        ax_eff_zoom.plot(etr_grid, eff_non, style, color=color)
    else:
        ax_eff.plot(etr_grid, eff, style, label=label, color=color)
        ax_eff_zoom.plot(etr_grid, eff, style, color=color)

    ax_eff.grid(True)


def plot_elas(df_b, formulas, axes=None, label="", color="gray", div=None, ref_etr=0.25, style='-', rate="etr", model="log"):
    """Plot elasticity implied by a marginal formula and scatter observed points.

    - df_b: dataframe with at least an `etr` column and optionally a boolean `haven`.
    - formulas: sequence where formulas[1] is the marginal expression evaluated
                to obtain elasticity-like values.
    - axes: optional axes or None; if None this function will plot to the
            current axes.
    """

    if div is None:
        div = [1, 0]

    sns.despine(bottom=True, left=True, right=True)
    plt.xlabel("$1 - \\tau $")
    plt.ylabel("Elasticity")

    if rate == "metr":
        etr_grid = 1 - np.linspace(0.015, 0.5, 1000)
    else:
        etr_grid = np.linspace(0.015, 0.5, 1000)

    # helper for safe eval
    def _eval(expr, val, haven_flag=None):
        local_ctx = {rate: val, "log": log}
        if haven_flag is not None:
            local_ctx["haven"] = 1 if haven_flag else 0
        return eval(expr, {}, local_ctx)

    # compute elasticity-like curves
    if "haven" in formulas[0]:
        eff_haven = np.array([_eval(formulas[1], v, haven_flag=1) for v in etr_grid]) + div[1]
        eff_haven = eff_haven * etr_grid
        eff_non = np.array([_eval(formulas[1], v, haven_flag=0) for v in etr_grid]) + div[1]
        eff_non = eff_non * etr_grid
    else:
        eff = np.array([_eval(formulas[1], v) for v in etr_grid])
        eff = eff * etr_grid

    # plotting
    if axes is None:
        ax = plt.gca()
    else:
        ax = axes

    if "haven" in formulas[0]:
        ax.plot(etr_grid, eff_haven, "--", label=label + ": Havens", color=color)
        ax.plot(etr_grid, eff_non, style, label=label + ": Non-havens", color=color)
        # scatter observed points
        if "haven" in df_b.columns:
            for etr in df_b.loc[df_b["haven"], "etr"]:
                ax.scatter(1 - etr, np.random.randn() * max(eff_haven) / 30 + (1 - etr) * _eval(formulas[1], 1 - etr, haven_flag=1),
                           color=color, s=10, alpha=0.5)
            for etr in df_b.loc[~df_b["haven"], "etr"]:
                ax.scatter(1 - etr, np.random.randn() * max(eff_non) / 30 + (1 - etr) * _eval(formulas[1], 1 - etr, haven_flag=0),
                           color=color, s=10, alpha=0.5)
    else:
        ax.plot(etr_grid, eff, style, label=label, color=color)
        for etr in df_b["etr"]:
            ax.scatter(1 - etr, np.random.randn() * max(eff) / 30 + (1 - etr) * _eval(formulas[1], 1 - etr), color=color, s=10, alpha=0.5)
    ax.grid(True)



def plots(df_b, prefix="",by_year=False, reform=lambda x: 2 if x<2014 else (0 if x<=2017 else 1), get_label=lambda x: ['2015-2017','2018-2020','2012-14'][x], add="", tax_havens=[]):
    """Run multiple model specifications, plot elasticity curves and cumulative residuals.

    Parameters
    - df_b: base DataFrame containing the variables used by the models
    - prefix: filename prefix for saved outputs
    - by_year: whether to run models by individual years or grouped periods
    - reform, get_label: helper callables to map year -> group and label
    - add: extra formula terms appended to model strings
    - tax_havens: list of iso codes to mark as havens

    Returns
    - summary table (pandas DataFrame) with model statistics (also saved to Excel)
    """
    fig1 = plt.figure(figsize=(12,6))
    results = []
    years = []
    if by_year:
        groups = sorted(df_b["year"].unique())
        def colors(year_val):
            return colors_l.pop(0)
        df_b["post_reform"] = df_b["year"]
        def get_label(year_val):
            return str(year_val)
    else:
        def colors(x):
            return "tomato" if x == 1 else ("cornflowerblue" if x == 0 else "gray")
        groups = [2,0,1]
        df_b["post_reform"] = df_b["year"].map(reform)
        
    
    df_b["haven"] = df_b["iso"].isin(tax_havens)
    df_b["metr"] = 1 - df_b["etr"]
    rate = "metr"
    models = [f"log(1.001-{rate}) +  {rate}", f"{rate}", f"{rate} + I({rate}**2)", f"haven*{rate}"]
    plt.figure(figsize=(10,6))
    model_title = ["Logarithmic", "Linear", "Quadratic", "Haven dummy"]
    model_names = []
    style = "-"
    for i,model in enumerate(models): 
        mt = model_title[i]
        ax = plt.subplot(2,2,i+1)
        ax_t = fig1.add_subplot(2,2,i+1)
        colors_l = list(sns.color_palette("tab10"))
        
        for year in groups:
            try:
                col = colors(year)
            except Exception:
                colors_l = list(sns.color_palette("tab20"))
                style = "--"
                
            data = df_b.loc[df_b["post_reform"]==year]
            if len(data) == 0:
                continue
            model_names.append(f"{mt} {get_label(year)}")
            data = data[["haven","pi",rate,"etr","statrate","t_at","wages","GDP_int","POP_int","year"]].dropna()


            *log_v,r1 = run_regression(data, f"log(pi) ~ {model} + log(t_at) + log(wages) + log(GDP_int) + log(POP_int) {add}", display_b=False,
                                      rate=rate, val="1.001 -")

            rates, resid = zip(*sorted(zip(data[rate], r1.resid))[::-1])
            ax_t.plot(rates, np.cumsum(resid), style, label=f"{get_label(year)}", color=col)

            results.append(r1)
            years.append(year)
            plot_elas(data, log_v,axes=ax,label=f"{get_label(year)}",style=style, color=col,model="log",rate=rate)

        plt.legend(frameon=False)
        plt.title(mt)

        ax_t.legend()
        ax_t.set_title(mt)
        ax_t.plot([0.5,1],[0,0])
        ax_t.set_xlabel("$1 - \\tau$")
        ax_t.set_ylabel("Cumulative residual")
        ax_t.grid(True)
        sns.despine(bottom=True,left=True,ax=ax_t)
        
    plt.tight_layout()

    plt.savefig(f"results/figures/{prefix}elasticities.pdf",bbox_inches="tight")
    fig1.set_tight_layout(True)
    fig1.savefig(f"results/figures/{prefix}cum_residual.pdf",bbox_inches="tight")


    summary = summary_col(results,stars=True,model_names=model_names,
                   info_dict={
            'N':lambda x: "{0:d}".format(int(x.nobs)),
            'R2':lambda x: "{:.2f}".format(x.rsquared),
            'BIC':lambda x: "{:.2f}".format(x.bic)
        }).tables[0]

    summary.index = [_.replace("metr", "(1-ETR)") for _ in summary.index]
    summary.index = [_.replace("1.001 - (1-ETR)", "0.001+ETR") for _ in summary.index]
    summary.index = [_.replace("[T.True]", "") for _ in summary.index]
    summary.to_excel(f"results/{prefix}elasticities.xlsx")
    return summary



def plot_compare(df,year,source1,source2,correct=False):
    """Compare profits from two sources for a given year and plot a scatter.

    Keeps previous behavior but uses clearer local variable names.

    Parameters
    - df: DataFrame containing columns ['year','source','type','iso','pi']
    - year: integer year to compare
    - source1, source2: strings identifying the two sources
    - correct: reserved compatibility flag (not used)
    """

    texts = []
    # Select and aggregate source1
    df_s1 = df.loc[(df["year"] == year) & (df["source"] == source1) & (df["type"] == "all_groups"), ["iso", "pi"]].copy()
    df_s1["iso"] = df_s1["iso"].replace("CYM", "UK_Caribbean").replace("VGB", "UK_Caribbean")
    df_s1 = df_s1.groupby("iso").sum().reset_index()

    # Merge with source2 series
    df_merged = pd.merge(
        df_s1,
        df.loc[(df["year"] == year) & (df["source"] == source2) & (df["type"] == "all_groups"), ["iso", "pi"]],
        on="iso",
    )
    df_merged["iso"] = df_merged["iso"].replace("UK_Caribbean", "UK-CAR")

    # Normalise and filter implausible values (preserve prior behavior)
    df_merged.loc[df_merged["pi_x"] < 0, "pi_x"] = 1e8
    df_merged.loc[df_merged["pi_y"] < 0, "pi_y"] = 1e8
    df_merged["pi_x"] = df_merged["pi_x"].fillna(-9)
    df_merged["pi_y"] = df_merged["pi_y"].fillna(-9)
    df_merged = df_merged.loc[(df_merged["pi_x"] > 0) & (df_merged["pi_y"] > 0)]

    corr = kendalltau(df_merged["pi_x"], df_merged["pi_y"])
    print(year, source1, source2, corr)

    for _, row in df_merged.iterrows():
        iso_code = row["iso"]
        if iso_code in ("USA", "Rest"):
            continue

        plt.scatter(row["pi_x"], row["pi_y"], color="gray")

        if np.abs(np.log2(row["pi_x"] / row["pi_y"])) > np.log2(2):
            if max([row["pi_x"], row["pi_y"]]) > 1e10:
                plt.scatter(row["pi_x"], row["pi_y"], color="olive")
                texts.append(plt.text(row["pi_x"], row["pi_y"], iso_code, color="olive"))
        elif np.abs(np.log2(row["pi_x"] / row["pi_y"])) > np.log2(1.5):
            texts.append(plt.text(row["pi_x"], row["pi_y"], iso_code))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(source_to_label[source1],fontsize=12)
    plt.ylabel(source_to_label[source2],fontsize=12)
    lims = [1E8,2E11]
    plt.plot(lims,lims,"--",color="darkgray",zorder=0)
    plt.grid()
    sns.despine(left=True,bottom=True)
    plt.xlim(lims)
    plt.ylim(lims)    
    plt.annotate(fr"Kendall $\tau$ = {corr[0]:0.2f}",(lims[0]*1.1,lims[1]*0.9), ha="left",va="top",fontsize=12)

    adjust_text(texts, df_merged["pi_x"].values, df_merged["pi_y"].values,
                arrowprops=dict(arrowstyle='->', color='gray'),
                   expand_points=(1.2,1.2),force_points=(0.6,0.9))
    


def plot_evol_variable(df,source,var="etr",ylabel="Effective tax rate (%)",letter="",flag_normalize=True,tax_havens=[]):
    """Plot the evolution of a variable (default `etr`) by country category.

    Parameters
    - df: DataFrame containing at least ['source','type','iso3','year','txc','pi', ...]
    - source: source string to filter the DataFrame
    - var: variable to plot (default 'etr')
    - ylabel: y-axis label
    - letter: optional panel letter (kept for compatibility)
    - flag_normalize: whether to normalize non-ETR series by year totals
    - tax_havens: list of iso3 codes to treat as tax havens

    This function preserves previous computation logic while clarifying
    variable naming for readability.
    """

    data = df.loc[df["source"] == source]
    if (var == "etr") and (source == "CBCR"):
        group_type = "positive_groups"
        data = data.loc[data["type"] == group_type]
    else:
        group_type = "all_groups"
        data = data.loc[data["type"] == group_type]
    
    data["1"] = 1
    rest = data.loc[data["iso3"]=="Rest"].sort_values(by="year")
    rest["category_iso"] = "Other countries"
    haven = data.loc[data["iso3"].isin(tax_havens)].groupby("year").sum().reset_index()
    haven["category_iso"] = "Tax havens"

    rest.loc[:,["txc","pi","emp","t_at","at"]] -= haven.loc[:,["txc","pi","emp","t_at","at"]].values
    us = data.loc[data["iso3"]=="USA"]
    us["category_iso"] = "Domestic"
    
    data = pd.concat([us,rest,haven])
    data["etr"] = 100*data["txc"]/data["pi"]
    
    pal = {"Tax havens": "coral",
          "Other countries":"chocolate",
          "Domestic":"royalblue",}

    if (var != "etr") and flag_normalize:
        data[var] = 100*data[var]/data.groupby("year")[var].transform(np.sum)
        

    for c,row in data.groupby("category_iso"):
        plt.plot(row["year"],row[var],"o-",color=pal[c],lw=2,label=c)
        for y,etr in zip(row["year"],row[var]):
            if (c == "Domestic"):# and (y != 2018):
                va = "top"
                offset = -0.5
            else:
                va = "bottom"
                offset = 0.5
            if "%" in ylabel:
                plt.annotate(f"{etr:2.1f}%",(y,etr+offset),ha="center",va=va,color=pal[c])
            else:
                plt.annotate(f"{etr:2.0f}",(y,etr+offset),ha="center",va=va,color=pal[c])
            

    
    plt.ylim(np.min([0,data[var].min()*1.2]),data[var].max()+1.5)
    plt.gca().grid(axis='y')
    sns.despine(bottom=False,left=True)
    plt.xticks(row["year"])
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend(frameon=False,loc=1, borderaxespad=0.)
    


