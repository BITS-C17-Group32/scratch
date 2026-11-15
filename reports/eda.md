# Exploratory Data Analysis: Demand Forecasting

## 1. Executive Overview & Dataset Architecture
This exploratory data analysis establishes the narrative and structural foundation for the machine learning demand forecasting system. The modeled environment is a fully balanced time-series grid encompassing **3,008,016 rows and 26 columns**, representing interactions across **1,688 dates**, **54 stores**, and **33 product families**. 

To maintain strict evaluation honesty, this analysis strictly limits its scope to the `train` (2,106,324 rows; 2013-01-01 to 2016-03-27) and `valid` (299,376 rows; 2016-03-28 to 2016-09-11) partitions. The `local_test` partition (602,316 rows) is entirely omitted from this exploration.

## 2. Macro Trend Analysis
An evaluation of multi-year sales growth from 2013 through late 2016 reveals a steady expansion in retail unit volume alongside a systemic contraction in zero-demand sparsity. 

![Figure 1: Annual Aggregate Unit Volume vs. Zero-Sales Row Contraction](figures/fig01_annual_trend_zero_share.png)
*Figure 1: Annual total units sold against the absolute count of zero-sales rows. Note that 2016 data is partial, representing transactions strictly through 2016-09-11.*

As observed in Figure 1, aggregate retail volume expanded from [FIG01_2013_UNITS] million units in 2013 to [FIG01_2015_UNITS] million units in 2015. Simultaneously, the share of zero-sales rows contracted, indicating expanding product catalog maturity and improved supply chain stocking reliability. 

## 3. Cyclical & Multi-Frequency Seasonality
Retail demand exhibits compounding cyclical behaviors driven by weekly shopping rhythms and bi-monthly macroeconomic liquidity events.

### Day-of-Week Shopping Rhythms
![Figure 2: Day-of-Week Seasonality Index](figures/fig02_weekly_seasonality_dow.png)
*Figure 2: Normalized daily sales index across the week, stratified by store format (A–E).*

Figure 2 demonstrates weekend-heavy retail patterns. The baseline index of 100 represents average daily volume. For example, Store Type A demonstrates a weekend surge reaching an index of 128.0 on Saturdays, confirming that predictive models must heavily weight categorical day-of-week embeddings to capture this variance.

### Bi-Monthly Payroll Spikes (Quincena)
![Figure 3: Intra-Month Bi-Monthly Payroll Demand](figures/fig03_payday_seasonality_bimonthly.png)
*Figure 3: Aggregate mean unit sales mapped against the 31-day calendar month, isolating the public sector payroll effect.*

Ecuadorian economic activity is heavily influenced by the public sector bi-monthly pay cycle (*quincena*). Figure 3 highlights distinct structural demand surges. Interestingly, the immediate 15th registers at an index of 97.6, while month-end closures hit 103.2 (Day 30). The true liquidity impact frequently spills into the immediate days following, such as Day 1 (Index: 111.7) and Day 31 (Index: 112.3).

## 4. Exogenous Shocks: 2016 Manabí Earthquake
On April 16, 2016, a devastating 7.8 Mw earthquake struck the Manabí region. Because this date sits fully inside the validation split boundary (2016-03-28 to 2016-09-11), it provides a crucial test of the models' ability to handle exogenous shock recovery.

![Figure 4: April 16, 2016 Manabí Earthquake Impact](figures/fig04_earthquake_impact_decay.png)
*Figure 4: Sales surge in emergency and essential staples following the April 16 earthquake, mapped against the 30-day linear decay feature.*

As shown in Figure 4, essential staples experienced a massive structural reallocation of demand, surging from a pre-quake mean of 466,794 units to a peak of 1,018,683 units (+118.2%). The engineered feature `earthquake_decay` maps this shock recovery dynamically, degrading from 1.0 (event day) to 0.0 over a rigid 30-day recovery window ending on 2016-05-15.

## 5. Calendar Leads & Promotional Uplift
National holidays severely distort localized baseline demand due to combined anticipation purchasing and day-of structural closures. 

![Figure 5: Holiday Lead Anticipation vs Event-Day Contraction](figures/fig05_holiday_lead_anticipation.png)
*Figure 5: Aggregate sales uplift percentages mapped against calendar lead indicators (is_holiday_tomorrow, is_holiday_in_2days).*

Figure 5 contrasts the predictive power of anticipation features against the actual event day. The day preceding a holiday (`is_holiday_tomorrow = 1`) generates an average volume uplift of +5.11%, whereas major national eves (like Dec 23-24) cause massive +53.07% surges. Conversely, synthetic Christmas closures result in a -100% contraction (0 units/day).

## 6. Store Life-Cycle & Structural Anomalies
Retail footprint expansion introduces massive blocks of non-stationary sparsity. Store 52 is the primary anomaly in the dataset.

![Figure 6: Store 52 Pre-Opening Zero Timeline](figures/fig06_store52_pre_opening_timeline.png)
*Figure 6: The inactive lifecycle of Store 52 from 2013-01-01 to 2016-09-11.*

Store 52 did not open for business until 2017-04-20. Consequently, it contributes **51,810 full-grid zeros** (and 12,804 holdout zeros) to the dataset. Within our permitted train and valid splits (2013-01-01 to 2016-09-11), it registers **44,550 unmasked zeros**. Because training a model on inactive store history suppresses baseline predictions, the data pipeline masks 44,451 of these pre-opening rows during training batches, retaining only 99 `id=NaN` synthetic Christmas closure rows as true-zero structural labels.

## 7. Cross-Sectional Revenue Dispersion
Demand is not distributed uniformly; it is highly concentrated both categorically and geographically.

![Figure 7: Product Family Pareto Concentration](figures/fig07_family_volume_dispersion.png)
*Figure 7: Cumulative volume distribution and sparsity share across the 33 product families.*

Figure 7 proves a stark Pareto concentration. The top 3 categories account for 62.43% of total volume, with GROCERY I alone dominating the mix at 32.82%. Conversely, long-tail categories like Books and Baby Care suffer from severe intermittency.

![Figure 8: Store Cluster & Geographic Demand Distribution](figures/fig08_cluster_geographic_sales.png)
*Figure 8: Average daily volume distributed by administrative store cluster and geographic state.*

Geographically (Figure 8), demand tracks metropolitan density. Store Cluster 5 moves 8.82x more daily volume (34,260 units/store/day) than the lowest-performing Cluster 7 (3,884 units/store/day). Additionally, the state of Pichincha commands 54.89% of the entire network's daily volume.

## 8. Demand Sparsity & Intermittency Topology
The most critical structural characteristic of this dataset is its zero-inflation. Across the entire 3,008,016-row balanced grid, approximately **~31.5%** of the rows represent zero unit sales. When rigorously measured strictly against our `train` and `valid` partitions, the empirical zero-sales share sits at exactly **35.2%**.

This severe zero-inflation dictates our modeling methodology: Mean Absolute Percentage Error (MAPE) is mathematically invalid due to pervasive division-by-zero errors. Consequently, the team exclusively relies on RMSLE, MAE, and RMSE to evaluate forecasting performance.

