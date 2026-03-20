# 📊 Data Science & GenAI Assessment

> Store sales forecasting with Random Forest (MAPE 6.54%) and a TF-IDF-based Policy Q&A bot — built using Python, scikit-learn, and Google Colab without any paid APIs.

**Name:** RAVINDRA KUMAR NAYAK &nbsp;|&nbsp; **Roll No.:** 2201MC30

---

## 📁 Repository Structure

```
├── Assignment1_Store_Sales_Analysis.ipynb   # Sales EDA, forecasting model & insights
├── Assignment2_Policy_QA_Bot.ipynb          # Policy Q&A bot using TF-IDF
├── sales_data.csv                           # Dataset for Assignment 1
├── leave_policy.txt                         # Leave policy document
├── it_policy.txt                            # IT policy document
├── travel_policy.txt                        # Travel policy document
└── README.md
```

---

## Assignment 1 — Store Sales Analysis & Prediction

### Objective
Analyze a full year of daily store sales data, engineer features, train a forecasting model, evaluate it on the last 30 days, predict the next 7 days, and extract business insights.

### Dataset
`sales_data.csv` — 366 rows covering the full year 2024.

| Column | Description |
|--------|-------------|
| `Date` | Calendar date (2024-01-01 to 2024-12-31) |
| `Sales` | Daily revenue in ₹ |
| `Promotion` | 1 = promotion ran that day, 0 = no promotion |
| `DayOfWeek` | 1 = Monday … 7 = Sunday |

### Approach

**EDA** → checked missing values, plotted sales over time, compared promotion vs non-promotion sales, and analyzed sales by day of week.

**Feature Engineering** → created lag features (`lag_1`), rolling averages (`rolling_avg_7`, `rolling_avg_14`), and calendar features (`Month`, `WeekOfYear`, `Quarter`, `IsWeekend`). All lag/rolling features use `.shift(1)` to prevent data leakage.

**Model** → Random Forest Regressor (200 trees, max depth 10). Train/test split is strictly chronological — last 30 days as test set, no random shuffling.

**Forecast** → Recursive 7-day forecasting: each prediction feeds back as `lag_1` for the next day.

### Results

| Metric | Value |
|--------|-------|
| MAE (Mean Absolute Error) | ₹ 12.67 |
| RMSE (Root Mean Squared Error) | ₹ 16.49 |
| MAPE (Mean Absolute % Error) | **6.54%** |
| 7-Day Forecast Total | ₹ 1,419.40 |

### Key Insights
1. **Promotions drive +23.9% sales lift** — running on only 19.4% of days but generating outsized revenue.
2. **Wednesday is the peak day** (₹ 227.66 avg) and **Saturday the weakest** (₹ 185.00 avg).
3. **Weekends underperform weekdays** by ₹ 27.72/day on average — unusual for retail.
4. **Promotion is the #1 model feature** with 50.2% importance — the single biggest business lever.
5. **Clear seasonal patterns** enable reliable advance planning for inventory and staffing.

---

## Assignment 2 — Simple Policy Q&A Bot

### Objective
Build a Q&A system that retrieves answers from three company policy documents using classical NLP — no paid APIs, no GPU-based models.

### Documents
| File | Content |
|------|---------|
| `leave_policy.txt` | Paid leave, sick leave, maternity, casual leave |
| `it_policy.txt` | VPN, passwords, laptop, antivirus, USB devices |
| `travel_policy.txt` | Flights, hotel limits, local travel, international approval |

### Approach — Two-Stage Retrieval

A single-stage TF-IDF approach was found to fail on ambiguous queries. For example, `"sick leaves allowed"` returned an IT Policy sentence because the word `"allowed"` scored higher in `"USB devices are not allowed..."` than in the correct Leave Policy sentence.

The fix is a **two-stage retrieval system**:

**Stage 1 — Domain Detection**
Count domain-specific keywords in the question to identify which policy it belongs to. If no keywords match any domain, return the fallback immediately — no TF-IDF is run.

```
Question: "sick leaves allowed"
  Leave Policy  → 2 matches  ('sick', 'leaves')
  IT Policy     → 0 matches
  Travel Policy → 0 matches
  → Route to Leave Policy ✅
```

**Stage 2 — TF-IDF + Cosine Similarity (within domain only)**
Convert the question to a TF-IDF vector and compare it only against sentences from the detected domain. Return the best match if the cosine similarity score exceeds 0.15, otherwise return the fallback message.

```
TF-IDF Config:
  stop_words    = 'english'
  ngram_range   = (1, 2)      # unigrams + bigrams
  threshold     = 0.15
  matrix size   = 17 sentences × 130 unique terms
```

### Results — 11 / 11 Test Cases Passed

| Question | Domain | Answer | Score |
|----------|--------|--------|-------|
| How many paid leaves per year? | Leave | Entitled to 15 paid leaves per year | 0.616 |
| What is maternity leave duration? | Leave | Maternity leave is 26 weeks | 0.621 |
| Can I carry forward unused leaves? | Leave | Cannot be carried forward | 0.513 |
| Is VPN required for remote work? | IT | VPN is mandatory for remote access | 0.338 |
| Can I use a USB drive at work? | IT | USB devices not allowed without IT approval | 0.258 |
| What device for official work? | IT | Company-issued laptop must be used | 0.507 |
| Hotel reimbursement limit per night? | Travel | Limit is Rs. 5000 per night | 0.739 |
| What class for flight bookings? | Travel | Must be economy class | 0.378 |
| Approval needed for international trip? | Travel | Requires prior managerial approval | 0.564 |
| Sick leaves allowed | Leave | Sick leave entitlement is 10 days | 0.191 |
| What is the canteen menu today? | None | Information not available in policy documents | N/A |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| matplotlib + seaborn | EDA visualizations |
| scikit-learn | RandomForestRegressor, TfidfVectorizer, cosine_similarity |
| Google Colab | Free CPU runtime — no GPU needed |

---

## 🚀 How to Run

1. Open the `.ipynb` files in [Google Colab](https://colab.research.google.com)
2. Upload the required data files when prompted (`sales_data.csv` for Assignment 1, the three `.txt` files for Assignment 2)
3. Run all cells top to bottom

> No additional setup required. All libraries are pre-installed on Google Colab.

---

## ⚙️ Constraints Respected
- ✅ No paid APIs used
- ✅ No GPU-based large models
- ✅ Free tools only (Google Colab, scikit-learn, pandas)
- ✅ Code runs without errors end to end
