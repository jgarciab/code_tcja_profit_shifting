Replication package for: "Did the Tax Cuts and Jobs Act Reduce Profit Shifting by US Multinational Companies?" (IMFE-D-24-00174R2)
Authors: Javier Garcia-Bernardo, Petr Janský, Gabriel Zucman


Quick summary
 - Process raw data (multiple sources) -> fix double counting -> produce figures and tables used in the manuscript.

Repository layout (key files)
 - `1a_processing_raw_data.ipynb` : Load raw sources, clean them, and build unified datasets used by the analysis.
	 - Note: Compustat data is not included in this replication package; see Data notes below.
	 - Output: cleaned, merged data placed under `data/`.

 - `1b_fix_double_counting.ipynb` : Apply fixes to remove double-counting issues across sources and produce the cleaned dataset used by analysis notebooks.
	 - Input: outputs from `1a`.

 - `2_figures_paper.ipynb` : Generate the main figures and visualizations used in the paper (plots aggregated from processed data).
	 - Input: cleaned data from `1b`.

 - `3. compare_irs.ipynb` : Compare effective tax rates (ETRs) and profit measures across different data sources; includes tables and diagnostic plots used in the manuscript.

 - `3a_robustness_data_processing.ipynb` and `3b_robusteness_figures_paper.ipynb` : Additional robustness processing and figures.

 - `f_helper.py` (inside this folder): Plotting and helper functions used by the notebooks (data-safe helpers, plotting wrappers, regression helpers).

 - `data_processing.py` (inside this folder): Data processing functions used by the notebooks (finding iso codes mostly).

Data
 - The `data_raw` contains the raw data files. `data/` directory contains processed TSV/CSV files used by the notebooks (e.g., `combined_dataset.tsv` and derivatives).
 - Important: Some proprietary data used in the analysis (e.g., full Compustat extracts) are NOT included in this replication package. Notebook code that references those sources will either skip or expect the user to supply them under `data/`.


Reproducing figures and tables
 - Directly run `2_figures_paper.ipynb` and `3b_robustness_figures_paper.ipynb`. Figures are saved into `results/figures/` 

Contact and citation
 - If you use this replication package, please cite the paper: "Did the Tax Cuts and Jobs Act Reduce Profit Shifting by US Multinational Companies?" (IMFE-D-24-00174R2) 
 - For questions, open an issue in the repository or contact the author(s) listed in the manuscript.


