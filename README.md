# M5 Retail Sales Forecasting: Deep Learning with Real-World Data Challenges

## Project Overview
This project focuses on building a multi-input Deep Learning model (specifically using LSTMs with embedding layers) to accurately forecast daily retail sales. The model was developed using the challenging Walmart M5 Forecasting Accuracy dataset, which presents significant complexities due to its large scale and time-series nature.

The primary goal was to develop a robust forecasting pipeline capable of identifying trends, seasonality, and the impact of external factors on sales, ultimately providing actionable insights for inventory and operational planning.

## Methodology & Key Features
* **Data Source:** Walmart M5 Forecasting Accuracy dataset (real-world retail sales, calendar, and pricing data).
* **Model Architecture:** Multi-input Keras model featuring:
    * LSTM layers for sequential sales data and engineered time-series features (lag values, rolling averages, seasonality indicators).
    * Embedding layers to capture intricate relationships within high-cardinality categorical IDs (e.g., item, department, store).
    * Dense layers for final sales prediction.
* **Feature Engineering:** Extensive time-series feature creation, including daily/weekly/monthly indicators, sales lags, and rolling mean/standard deviation.
* **Data Preprocessing:** Robust handling of missing values, meticulous data type optimization, and chronological train/validation/test splitting crucial for time-series forecasting.

## Overcoming Challenges & Key Learnings

This project served as an intensive masterclass in real-world data engineering and debugging, where I encountered and systematically resolved several critical issues:

1.  **Massive Data Scale & Memory Management:**
    * **Challenge:** The original ~58 million row dataset quickly led to Out-of-Memory (OOM) errors during initial loading and merging, even on cloud environments like Google Colab.
    * **Solution:** Implemented **aggressive dataset subsetting** by filtering for specific items and stores at the earliest possible stage in the pipeline. This ensured all subsequent memory-intensive operations ran on a manageable subset (~19,000 rows), enabling successful development.
    * **Learning:** Emphasized the critical importance of strategic **early data filtering** and **resource optimization** when dealing with big data on constrained hardware.

2.  **Persistent Data Type & Shape Mismatches:**
    * **Challenge:** Repeated `ValueError: Invalid dtype: object` errors during model training, and baffling sample count mismatches (e.g., `(15020,)` vs. `(30040,1)` for seemingly identical data lists). This implied non-numeric data or structural issues in the final NumPy arrays fed to TensorFlow.
    * **Solution:** Conducted deep-dive diagnostics (with custom print statements) revealing subtle behaviors:
        * NumPy sometimes converting lists of single-element arrays (e.g., `np.array([0])`) into unexpected structures, requiring explicit scalar extraction (`int(value)`) during list appending.
        * Original string ID columns inadvertently slipping into numerical `sequential_features` lists.
    * **Learning:** Reinforced the absolute necessity of **meticulous data type management** and **explicit type casting** (`np.float32`, `np.int32`) for every NumPy array used in Deep Learning. Every element's scalar nature must be guaranteed.

3.  **Pipeline Debugging & Resilience:**
    * **Challenge:** The cascading nature of errors meant a single fix could expose a new issue down the line, leading to numerous **kernel restarts and full pipeline re-runs**.
    * **Solution:** Adopted a **systematic debugging** approach, adding precise diagnostic checks at each stage to isolate the exact point of failure. Leveraged **Google Colab** to provide a stable environment for iteration.
    * **Learning:** Problem-solving in data science is an iterative process requiring exceptional **persistence** and **resilience**. The ability to methodically break down complex, multi-stage issues is paramount.

## Results
The trained Deep Learning model achieved a **Mean Absolute Error (MAE) of ~1.57 units** on the unseen test set for the selected items, demonstrating strong forecasting capability. Visualizations confirmed the model's ability to capture trends and predict sales effectively over time.

### Visualizations:
*(Here, you will embed screenshots of your plots. After uploading your images to your GitHub repository, you can get their raw URL and use Markdown syntax. For example: `![Actual vs Predicted Sales Plot](path/to/your/plot_screenshot.png)`)*

## Technologies Used
* **Languages:** Python
* **Deep Learning:** TensorFlow, Keras (LSTM, Embedding Layers)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (MinMaxScaler)
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab

## How to Run This Project
This project is best run in **Google Colab** due to its computational demands (even with subsetting) and to ensure environment consistency.

1.  **Download the Notebook:** Download the `M5_Forecasting_Analysis.ipynb` file from this repository.
2.  **Upload Data to Google Drive:** Download the [M5 Forecasting - Accuracy dataset from Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data). Extract the CSVs and upload the entire `data` folder directly into your Google Drive's "My Drive" root.
3.  **Open in Colab:** Go to [colab.research.google.com](https://colab.research.google.com/), click "File" -> "Upload notebook", and select the `M5_Forecasting_Analysis.ipynb` file.
4.  **Mount Drive:** In the first cell of the Colab notebook, ensure `from google.colab import drive; drive.mount('/content/drive')` is present and run it to connect to your Drive.
5.  **Set Data Path:** Update the `DATA_PATH` variable in the first cell to `DATA_PATH = '/content/drive/MyDrive/data/'`.
6.  **Set Runtime Type:** Go to `Runtime` -> `Change runtime type` and select `GPU` (e.g., `T4 GPU`).
7.  **Run All Cells:** Execute all cells sequentially (`Runtime` -> `Run all`).

---
