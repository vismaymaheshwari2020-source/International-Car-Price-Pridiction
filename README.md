# International Car Price Prediction

An end-to-end Machine Learning and Data Science project to analyze vehicle specifications and predict used car market valuations across international automotive markets using advanced regression models.

---

## 📌 Project Overview

Vehicle pricing is governed by a multitude of intersecting attributes including manufacturing era, mechanical condition, engine volume, fuel source, transmission technology, drivetrain architecture, and brand pedigree.

The **International Car Price Prediction** project leverages a comprehensive dataset of **56,244 vehicle listings** across 12 feature dimensions. Through rigorous Exploratory Data Analysis (EDA), feature engineering, and supervised learning, this project identifies market valuation trends and constructs a predictive Decision Tree Regressor achieving an **85.29% accuracy score ($R^2 \approx 0.85$)**.

---

## 📊 Dataset & Data Dictionary

The dataset is stored in [`cars.csv`](cars.csv) and contains 56,244 observations:

| Feature | Description | Type |
|---|---|---|
| `make` | Vehicle brand / manufacturer | Categorical |
| `model` | Specific vehicle model name | Categorical |
| `priceUSD` | Target variable: Market price in USD ($) | Continuous |
| `year` | Year of vehicle manufacture | Continuous |
| `condition` | Operational state at time of listing (with mileage, damaged, for parts) | Categorical |
| `mileage(kilometers)` | Total distance travelled in kilometers | Continuous |
| `fuel_type` | Fuel configuration (`petrol`, `diesel`, `electro`) | Categorical |
| `volume(cm3)` | Engine displacement volume in cubic centimeters ($\text{cm}^3$) | Continuous |
| `color` | Exterior body color | Categorical |
| `transmission` | Gearbox mechanism (`manual`, `auto`) | Categorical |
| `drive_unit` | Drivetrain layout (`front-wheel drive`, `rear-wheel drive`, `all-wheel drive`, `part-time four-wheel drive`) | Categorical |
| `segment` | Market segment class | Categorical |

---

## 🔍 Key Exploratory Data Analysis (EDA) Insights

1. **Manufacturing Era Trends**: Vehicle prices remained relatively steady between 1980 and 2000, followed by exponential valuation growth in modern vehicles produced after 2005.
2. **Fuel Type Valuation**: Traditional petrol and diesel vehicles show comparable mileage distributions, but modern petrol vehicles command a notable premium over diesel counterparts. Electric vehicles (`electro`) represent the highest valuation bracket on average.
3. **Transmission Dynamics**: Automatic transmission vehicles exhibit substantially higher resale prices than manual transmission vehicles, with a widening divergence post-2000.
4. **Drivetrain Premium**: All-Wheel Drive (AWD) vehicles consistently maintain the highest resale value among all drive architectures, followed by part-time four-wheel drive and rear-wheel drive.
5. **Brand Segment Hierarchy**: Specialty vehicles (e.g., Porsche, Bentley, Maserati, McLaren, Tesla) hold the highest market value, closely followed by Luxury European, American, Asian, and Mainstream European segments.

---

## 🛠️ Data Preprocessing & Feature Engineering Pipeline

- **Brand Segment Categorization**: High-cardinality `make` values were clustered into macroeconomic brand tiers:
  - *Luxury European* (Audi, BMW, Jaguar, Land-Rover, Alfa-Romeo, etc.)
  - *Mainstream European* (Volkswagen, Renault, Volvo, Fiat, Skoda, Opel, etc.)
  - *Russian / Eastern European* (Lada-VAZ, GAZ, UAZ, Moskvich, etc.)
  - *Asian* (Toyota, Honda, Nissan, Hyundai, Kia, Lexus, etc.)
  - *American* (Ford, Chevrolet, Cadillac, Jeep, Buick, GMC, etc.)
  - *Specialty* (Porsche, Bentley, Maserati, Tesla, McLaren)
  - *Other*
- **Temporal Filtering**: Restricted to post-1980 models (`year > 1980`) to eliminate historical non-representative skew.
- **Missing Value Handling**: Incomplete entries dropped given large sample size integrity.
- **Label Encoding**: Structured categorical dimensions encoded via Scikit-Learn `LabelEncoder`.
- **Outlier Removal**: Statistical outlier rejection applied via Z-score thresholding ($|Z| < 3$) across continuous variables (`year`, `mileage(kilometers)`, `volume(cm3)`).

---

## 📈 Model Architecture & Performance

A **Decision Tree Regressor** was tuned using 5-Fold Cross-Validated **GridSearchCV** over hyperparameters (`max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`):

- **Model**: Decision Tree Regressor
- **Best Hyperparameters**:
  - `max_depth`: 8
  - `min_samples_leaf`: 4
  - `min_samples_split`: 2
  - `random_state`: 0
- **Test Set Performance**:
  - **$R^2$ Score**: **0.8529 (85.29%)**
  - **Most Influential Predictive Features**:
    1. `year` (Year of Production) — Dominant price determinant
    2. `volume(cm3)` (Engine Displacement) — Key power/class indicator
    3. `drive_unit` & `transmission` — Key structural differentiators

---

## 📂 Project Structure

```
International Car Price Prediction/
├── International Car Price Prediction.ipynb   # Complete interactive Jupyter Notebook with outputs
├── International Car Price Prediction.pdf     # High-resolution PDF document export
├── cars.csv                                   # Complete vehicle dataset (56,244 records)
├── description.md                             # Summary overview & data dictionary
├── train_model.py                             # Python automation pipeline script
├── requirements.txt                           # Python dependencies
└── README.md                                  # Comprehensive project documentation
```

---

## 🚀 Getting Started

### 1. Environment Setup

Clone or place the project folder into your workspace, and install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Jupyter Notebook

Open and execute the notebook in Jupyter Lab or Jupyter Notebook:

```bash
jupyter notebook "International Car Price Prediction.ipynb"
```

### 3. Run the Standalone Pipeline Script

Execute the training and evaluation pipeline directly from the terminal:

```bash
python train_model.py
```
