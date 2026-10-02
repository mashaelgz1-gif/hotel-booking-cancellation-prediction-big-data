
"""hotel-booking.ipynb


# 1. Set-up
"""

!pip -q install pyspark

# Commented out IPython magic to ensure Python compatibility.
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, sum, desc, avg, to_date, concat, lit, lpad
# %matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
import pyspark.sql.functions as sql_f
import numpy as np
from pyspark.sql import Row
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

spark = SparkSession.builder \
    .appName("HotelEDA") \
    .getOrCreate()

print("Spark ready ✅")

"""# 2. Loading Data"""

from google.colab import files
files.upload()

df = spark.read.csv("hotel_bookings.csv", header=True, inferSchema=True)

# use cache to improve performance
df.cache()

df.show(5)

df.printSchema()

"""# 3. Data Understanding

This step is for exploring the structure of the dataset.
We check the number of rows and columns, data types, and generate summary statistics to understand the data.
"""

print("Number of rows:", df.count())
print("Number of columns:", len(df.columns))

df.describe().show()

missing_df = df.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])

missing_df.show(truncate=False)

"""**Data description**

This dataset contains hotel booking information, including details about reservations, customer behavior, and booking outcomes.

**Attribute information**:

- **hotel**: Type of hotel (Resort Hotel or City Hotel)
- **is_canceled**: Booking cancellation status (1 = canceled, 0 = not canceled)
- **lead_time**: Number of days between booking and arrival
- **arrival_date_year**: Year of arrival
- **arrival_date_month**: Month of arrival
- **arrival_date_week_number**: Week number of arrival
- **arrival_date_day_of_month**: Day of arrival
- **stays_in_weekend_nights**: Number of weekend nights stayed
- **stays_in_week_nights**: Number of weekday nights stayed
- **adults**: Number of adults
- **children**: Number of children
- **babies**: Number of babies
- **meal**: Type of meal booked
- **country**: Country of origin of the customer
- **market_segment**: Market segment designation
- **distribution_channel**: Booking distribution channel
- **is_repeated_guest**: Whether the customer is a repeated guest (1 = yes, 0 = no)
- **previous_cancellations**: Number of previous canceled bookings
- **previous_bookings_not_canceled**: Number of previous successful bookings
- **reserved_room_type**: Type of room reserved
- **assigned_room_type**: Type of room assigned
- **booking_changes**: Number of changes made to the booking
- **deposit_type**: Type of deposit made
- **agent**: ID of the booking agent
- **company**: ID of the company that made the booking
- **days_in_waiting_list**: Number of days in the waiting list
- **customer_type**: Type of customer
- **adr**: Average Daily Rate (average price per night)
- **required_car_parking_spaces**: Number of parking spaces required
- **total_of_special_requests**: Number of special requests made by the customer
- **reservation_status**: Final status of the reservation
- **reservation_status_date**: Date of the reservation status

**Our target variable is:**

**is_canceled**: Indicates whether the booking was canceled (1) or not (0).

# 4. Exploratory Data Analysis (EDA)

In this phase, we analyze the distribution of data, and explore relationships between different features to gain insights and detect patterns.

## 4.1 Numeric Variables Analysis

### 4.1.1 Target Variable
"""

df.groupBy("is_canceled").count().show()

target_counts = df.groupBy("is_canceled").count().collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [(label_map[row["is_canceled"]], row["count"]) for row in target_counts]

(x_values, y_values) = zip(*data)

plt.figure(figsize=(7, 4))
plt.bar(x_values, y_values)
plt.title("Cancellation Distribution")
plt.xlabel("Booking Status")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

"""The distribution shows that the majority of bookings were not canceled, while a significant portion were canceled. This indicates that although cancellations are common, most customers tend to complete their bookings, resulting in a moderately imbalanced target variable.

### 4.1.2 Lead Time vs Cancellation
"""

df.select("lead_time").describe().show()
df.groupBy("is_canceled").avg("lead_time").show()

lead_data = df.groupBy("is_canceled").avg("lead_time").collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [(label_map[row["is_canceled"]], row["avg(lead_time)"]) for row in lead_data]

(x, y) = zip(*data)

plt.figure(figsize=(7, 4))
plt.bar(x, y)
plt.title("Lead Time vs Cancellation")
plt.xlabel("Booking Status")
plt.ylabel("Average Lead Time")
plt.tight_layout()
plt.show()

"""Bookings with longer lead times are more likely to be canceled, indicating a strong relationship between early reservations and cancellation behavior.

### 4.1.3 Average Daily Rate by Cancellation
"""

df.select("adr").describe().show()
df.groupBy("hotel").avg("adr").show()

adr_data = df.groupBy("is_canceled").avg("adr").collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [(label_map[row["is_canceled"]], row["avg(adr)"]) for row in adr_data]

(x, y) = zip(*data)

plt.figure(figsize=(7, 4))
plt.bar(x, y)
plt.title("Average Daily Rate by Cancellation")
plt.xlabel("Booking Status")
plt.ylabel("Average ADR")
plt.tight_layout()
plt.show()

"""Even though the difference is not very large, canceled bookings tend to have slightly higher prices, indicating that more expensive reservations may be more likely to be canceled.

### 4.1.4 Special Requests vs Cancellation
"""

df.groupBy("total_of_special_requests").count().show()

req_data = df.groupBy("is_canceled").avg("total_of_special_requests").collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [(label_map[row["is_canceled"]], row["avg(total_of_special_requests)"]) for row in req_data]

(x, y) = zip(*data)

plt.figure(figsize=(7, 4))
plt.bar(x, y)
plt.title("Special Requests vs Cancellation")
plt.xlabel("Booking Status")
plt.ylabel("Average Requests")
plt.tight_layout()
plt.show()

"""Customers with more special requests are less likely to cancel their bookings.

## 4.2 Categorical Variables Analysis

### 4.2.1 Cancellation by Hotel Type
"""

df.groupBy("hotel").count().show()
df.groupBy("hotel", "is_canceled").count().show()

hotel_cancel = df.groupBy("hotel", "is_canceled").count().collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [((row["hotel"], row["is_canceled"]), row["count"]) for row in hotel_cancel]

labels = [f"{h} - {label_map[c]}" for (h, c), _ in data]
values = [v for _, v in data]

plt.figure(figsize=(10, 4))
plt.bar(labels, values)
plt.title("Cancellation by Hotel Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""The results show that City Hotels have a higher number of both canceled and non-canceled bookings compared to Resort Hotels. Additionally, cancellations appear to be more frequent in City Hotels, suggesting that booking behavior may vary depending on the hotel type.

### 4.2.2 Bookings per Month
"""

df.groupBy("arrival_date_month").count().orderBy("count", ascending=False).show()

month_order = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

month_data = df.groupBy("arrival_date_month").count().collect()

data = sorted(
    [(row["arrival_date_month"], row["count"]) for row in month_data],
    key=lambda x: month_order.index(x[0])
)

(x, y) = zip(*data)

plt.figure(figsize=(12, 4))
plt.bar(x, y)
plt.title("Bookings per Month")
plt.xlabel("Month")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""The distribution of bookings varies across different months, with certain months showing higher booking activity than others. This indicates the presence of seasonal patterns in hotel demand, where bookings tend to increase during peak periods.

### 4.2.3 Customer Type vs Cancellation
"""

df.groupBy("customer_type").count().show()

cust_data = df.groupBy("customer_type", "is_canceled").count().collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [
    (f"{row['customer_type']} - {label_map[row['is_canceled']]}", row["count"])
    for row in cust_data
]

(x, y) = zip(*data)

plt.figure(figsize=(10, 4))
plt.bar(x, y)
plt.xticks(rotation=45)
plt.title("Customer Type vs Cancellation")
plt.tight_layout()
plt.show()

"""The results show that Transient customers represent the largest portion of bookings, with a high number of both canceled and non-canceled reservations. In contrast, Contract and Group customers have significantly fewer bookings and lower cancellation counts.

### 4.2.4 Deposit Type vs Cancellation
"""

df.groupBy("deposit_type").count().show()

dep_data = df.groupBy("deposit_type", "is_canceled").count().collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [
    (f"{row['deposit_type']} - {label_map[row['is_canceled']]}", row["count"])
    for row in dep_data
]

(x, y) = zip(*data)

plt.figure(figsize=(10, 4))
plt.bar(x, y)
plt.xticks(rotation=45)
plt.title("Deposit Type vs Cancellation")
plt.tight_layout()
plt.show()

"""The results show that bookings with a 'No Deposit' policy have the highest number of both canceled and non-canceled reservations. In contrast, 'Non Refund' bookings tend to have fewer cancellations, indicating that stricter payment policies reduce the likelihood of cancellation.

### 4.2.5 Market Segment vs Cancellation
"""

df.groupBy("market_segment").count().show()

seg_data = df.groupBy("market_segment", "is_canceled").count().collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [
    (f"{row['market_segment']} - {label_map[row['is_canceled']]}", row["count"])
    for row in seg_data
]

(x, y) = zip(*data)

plt.figure(figsize=(12, 5))
plt.bar(x, y)
plt.xticks(rotation=60)
plt.title("Market Segment vs Cancellation")
plt.tight_layout()
plt.show()

"""The results show that the Online Travel Agency (Online TA) segment has the highest number of bookings and cancellations compared to other segments. This suggests that the booking channel plays an important role in customer behavior and cancellation patterns.

### 4.2.6 Country vs Cancellation
"""

df.groupBy("country").count().orderBy("count", ascending=False).show(10)

country_cancel = df.groupBy("country", "is_canceled") \
    .count() \
    .orderBy("count", ascending=False) \
    .limit(10) \
    .collect()

label_map = {0: "Not Canceled", 1: "Canceled"}

data = [
    (f"{row['country']} - {label_map[row['is_canceled']]}", row["count"])
    for row in country_cancel
]

(x, y) = zip(*data)

plt.figure(figsize=(12, 5))
plt.bar(x, y)
plt.xticks(rotation=60)
plt.title("Top Countries vs Cancellation")
plt.tight_layout()
plt.show()

"""The results show that Portugal (PRT) has the highest number of bookings and cancellations among all countries. Other countries such as the United Kingdom (GBR), France (FRA), and Spain (ESP) have moderate booking activity with relatively lower cancellation counts.

# 5. Outliers
"""

numeric_cols_outlier = [c for c, dtype in df.dtypes if dtype in ['int', 'double', 'float', 'bigint']]

for c in numeric_cols_outlier:
    plt.figure(figsize=(8, 4))
    df.select(c).dropna().toPandas().boxplot()
    plt.title(f"Outliers: {c}")
    plt.tight_layout()
    plt.show()

"""The boxplots reveal that several numeric features such as lead_time, adr, and previous_cancellations contain significant outliers. These extreme values may affect model performance and should be considered during the preprocessing stage.

# 6. Trends

This section presents the main trends identified in the hotel booking dataset using PySpark.
"""

# 1. Booking Trend Over Time
df = df.withColumn(
    "arrival_date",
    to_date(
        concat(
            col("arrival_date_year").cast("string"), lit("-"),
            col("arrival_date_month"), lit("-"),
            lpad(col("arrival_date_day_of_month").cast("string"), 2, "0")
        ),
        "yyyy-MMMM-dd"
    )
)

trend_time = df.groupBy("arrival_date").count().withColumnRenamed("count", "bookings")
trend_time.orderBy("arrival_date").show()

trend_pd = trend_time.orderBy("arrival_date").toPandas()

plt.figure(figsize=(12, 4))
plt.plot(trend_pd["arrival_date"], trend_pd["bookings"])
plt.title("Booking Trend Over Time", fontsize=10)
plt.xlabel("Date")
plt.ylabel("Bookings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("fig6.png", dpi=300)
plt.show()

# 2. Lead Time vs Cancellation Rate
lead_trend = df.groupBy("lead_time") \
    .agg(avg("is_canceled").alias("cancellation_rate"))

lead_trend.orderBy("lead_time").show()

lead_pd = lead_trend.orderBy("lead_time").toPandas()

plt.figure(figsize=(8, 4))
plt.plot(lead_pd["lead_time"], lead_pd["cancellation_rate"])
plt.title("Lead Time vs Cancellation Rate", fontsize=10)
plt.xlabel("Lead Time")
plt.ylabel("Cancellation Rate")
plt.tight_layout()
plt.savefig("fig7.png", dpi=300)
plt.show()

# 3. Market Segment Cancellation Rate Over Years
market_year_trend = df.groupBy("arrival_date_year", "market_segment") \
    .agg(avg("is_canceled").alias("cancellation_rate")) \
    .orderBy("arrival_date_year", "market_segment")

market_year_trend.show()

market_pd = market_year_trend.toPandas()

plt.figure(figsize=(12, 5))

for segment in market_pd["market_segment"].unique():
    segment_data = market_pd[market_pd["market_segment"] == segment]
    plt.plot(segment_data["arrival_date_year"],
             segment_data["cancellation_rate"],
             marker="o", label=segment)

plt.title("Market Segment Cancellation Rate Over Years")
plt.xlabel("Year")
plt.ylabel("Cancellation Rate")
plt.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("market_segment_trend.png", dpi=300)
plt.show()

# 4. Customer Type Cancellation Rate Over Years
customer_year_trend = df.groupBy("arrival_date_year", "customer_type") \
    .agg(avg("is_canceled").alias("cancellation_rate")) \
    .orderBy("arrival_date_year", "customer_type")

customer_year_trend.show()

customer_pd = customer_year_trend.toPandas()

plt.figure(figsize=(10, 5))

for ctype in customer_pd["customer_type"].unique():
    ctype_data = customer_pd[customer_pd["customer_type"] == ctype]
    plt.plot(ctype_data["arrival_date_year"],
             ctype_data["cancellation_rate"],
             marker="o", label=ctype)

plt.title("Customer Type Cancellation Rate Over Years")
plt.xlabel("Year")
plt.ylabel("Cancellation Rate")
plt.legend()
plt.tight_layout()
plt.savefig("customer_type_trend.png", dpi=300)
plt.show()

# 5. Average ADR Trend Over Years
adr_trend = df.groupBy("arrival_date_year") \
    .agg(avg("adr").alias("avg_price"))

adr_trend.orderBy("arrival_date_year").show()

adr_pd = adr_trend.orderBy("arrival_date_year").toPandas()

plt.figure(figsize=(8, 4))
plt.plot(adr_pd["arrival_date_year"], adr_pd["avg_price"], marker="o")
plt.title("Average ADR Trend", fontsize=10)
plt.xlabel("Year")
plt.ylabel("Average Price")
plt.tight_layout()
plt.savefig("fig10.png", dpi=300)
plt.show()

"""# 7. Feature Selection"""

feature_cols = [
    "lead_time",
    "adr",
    "adults",
    "children",
    "babies",
    "booking_changes",
    "previous_cancellations",
    "total_of_special_requests"
]

label_col = "is_canceled"

df_ml = df.select(feature_cols + [label_col])

"""# 8. Data Cleaning and Transformation for ML"""

# Replace "NA" string with null
df_ml = df_ml.replace("NA", None)

# Cast all columns to double
for c in feature_cols + [label_col]:
    df_ml = df_ml.withColumn(c, col(c).cast("double"))

# Drop rows with null values
df_ml = df_ml.dropna(subset=feature_cols + [label_col])

# Verify no nulls remain
df_ml.select([
    count(when(col(c).isNull(), c)).alias(c)
    for c in df_ml.columns
]).show()

"""# 9. Correlation Matrix"""

corr_cols = feature_cols + [label_col]

corr_pd = df_ml.select(corr_cols).toPandas()

plt.figure(figsize=(10, 6))
sns.heatmap(
    corr_pd.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig("correlation_matrix.png", dpi=300)
plt.show()

"""# 10. VectorAssembler"""

vectorAssembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="skip"
)

df_ml = vectorAssembler.transform(df_ml)
df_ml = df_ml.select("features", label_col)

df_ml.show(5, truncate=False)

"""# 11. Train-Test Split"""

seed = 123456

df_train, df_test = df_ml.randomSplit([0.7, 0.3], seed=seed)

print("Training rows:", df_train.count())
print("Testing rows: ", df_test.count())

"""# 12. Model Building"""

# Logistic Regression
lr = LogisticRegression(
    labelCol="is_canceled",
    featuresCol="features"
)

# Decision Tree
dt = DecisionTreeClassifier(
    labelCol="is_canceled",
    featuresCol="features"
)

# Random Forest
rf = RandomForestClassifier(
    labelCol="is_canceled",
    featuresCol="features",
    numTrees=50
)



"""# 13. Model Training Without Cross Validation"""

model_lr = lr.fit(df_train)
model_dt = dt.fit(df_train)
model_rf = rf.fit(df_train)

pred_lr = model_lr.transform(df_test)
pred_dt = model_dt.transform(df_test)
pred_rf = model_rf.transform(df_test)

pred_lr.select(label_col, "prediction", "probability").show(5)
pred_dt.select(label_col, "prediction").show(5)
pred_rf.select(label_col, "prediction", "probability").show(5)

models_predictions_no_cv = {
    "Logistic Regression": pred_lr,
    "Decision Tree": pred_dt,
    "Random Forest": pred_rf
}

metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]

results_no_cv = {}

for model_name, predictions in models_predictions_no_cv.items():
    print("\n" + "="*40)
    print(model_name + " - Without CV")
    print("="*40)

    results_no_cv[model_name] = {}

    for metric in metrics:
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric
        )

        score = evaluator.evaluate(predictions)
        results_no_cv[model_name][metric] = round(score, 4)
        print(f"{metric:20s}: {score:.4f}")

# The extra Spark table confusion matrix was removed.
# The heatmap confusion matrix below is clearer and more useful for the report.

import numpy as np

for model_name, predictions in models_predictions_no_cv.items():
    cm_data = predictions.groupBy(label_col, "prediction").count().toPandas()

    cm = np.zeros((2, 2), dtype=int)
    for _, row in cm_data.iterrows():
        cm[int(row[label_col])][int(row["prediction"])] = row["count"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Canceled", "Canceled"],
        yticklabels=["Not Canceled", "Canceled"]
    )
    plt.title(f"Confusion Matrix Without CV - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()

"""# 14. Model Training with Cross Validation"""

evaluator = MulticlassClassificationEvaluator(
    labelCol="is_canceled",
    predictionCol="prediction",
    metricName="accuracy"
)

# --- Logistic Regression CV ---
paramGrid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1]) \
    .build()

cv_lr = CrossValidator(
    estimator=lr,
    evaluator=evaluator,
    estimatorParamMaps=paramGrid_lr,
    numFolds=3
)

pipeline_lr = Pipeline(stages=[cv_lr])
model_lr = pipeline_lr.fit(df_train)
print("Logistic Regression trained")

# --- Decision Tree CV ---
paramGrid_dt = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10]) \
    .build()

cv_dt = CrossValidator(
    estimator=dt,
    evaluator=evaluator,
    estimatorParamMaps=paramGrid_dt,
    numFolds=3
)

pipeline_dt = Pipeline(stages=[cv_dt])
model_dt = pipeline_dt.fit(df_train)
print("Decision Tree trained")

# --- Random Forest CV ---
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 50]) \
    .build()

cv_rf = CrossValidator(
    estimator=rf,
    evaluator=evaluator,
    estimatorParamMaps=paramGrid_rf,
    numFolds=3
)

pipeline_rf = Pipeline(stages=[cv_rf])
model_rf = pipeline_rf.fit(df_train)
print("Random Forest trained")

"""# 15. Model Prediction"""

pred_lr = model_lr.transform(df_test)
pred_dt = model_dt.transform(df_test)
pred_rf = model_rf.transform(df_test)

pred_lr.select(label_col, "prediction", "probability").show(5)
pred_dt.select(label_col, "prediction").show(5)
pred_rf.select(label_col, "prediction", "probability").show(5)

"""# 16. Performance Evaluation"""

models_predictions = {
    "Logistic Regression": pred_lr,
    "Decision Tree": pred_dt,
    "Random Forest": pred_rf
}

metrics = ["accuracy", "weightedPrecision", "weightedRecall", "f1"]

results = {}

for model_name, predictions in models_predictions.items():
    print("\n" + "="*40)
    print(model_name + " - With CV")
    print("="*40)

    results[model_name] = {}

    for metric in metrics:
        eval_metric = MulticlassClassificationEvaluator(
            labelCol=label_col,
            predictionCol="prediction",
            metricName=metric
        )
        score = eval_metric.evaluate(predictions)
        results[model_name][metric] = round(score, 4)
        print(f"{metric:20s}: {score:.4f}")

import pandas as pd

results_df = pd.DataFrame(results).T
results_df.columns = ["Accuracy", "Precision", "Recall", "F1"]

print("=" * 63)
print(f"{'Model':<25} {'Accuracy':>8} {'Precision':>9} {'Recall':>8} {'F1':>7}")
print("=" * 63)
for model, row in results_df.iterrows():
    print(f"{model:<25} {row['Accuracy']:>8} {row['Precision']:>9} {row['Recall']:>8} {row['F1']:>7}")
print("=" * 63)

"""# 16.1 Comparison: Without CV vs With CV"""

comparison = {
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy (No CV)": [
        results_no_cv["Logistic Regression"]["accuracy"],
        results_no_cv["Decision Tree"]["accuracy"],
        results_no_cv["Random Forest"]["accuracy"]
    ],
    "Accuracy (CV)": [
        results["Logistic Regression"]["accuracy"],
        results["Decision Tree"]["accuracy"],
        results["Random Forest"]["accuracy"]
    ],
    "F1 (No CV)": [
        results_no_cv["Logistic Regression"]["f1"],
        results_no_cv["Decision Tree"]["f1"],
        results_no_cv["Random Forest"]["f1"]
    ],
    "F1 (CV)": [
        results["Logistic Regression"]["f1"],
        results["Decision Tree"]["f1"],
        results["Random Forest"]["f1"]
    ]
}

comp_df = pd.DataFrame(comparison)

print("\n" + "=" * 75)
print("Comparison: Without CV vs With CV")
print("=" * 75)
print(comp_df.to_string(index=False))
print("=" * 75)

"""# 17. Confusion Matrix"""

for model_name, predictions in models_predictions.items():
    cm_data = predictions.groupBy(label_col, "prediction").count().toPandas()

    cm = np.zeros((2, 2), dtype=int)
    for _, row in cm_data.iterrows():
        cm[int(row[label_col])][int(row["prediction"])] = row["count"]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Canceled", "Canceled"],
        yticklabels=["Not Canceled", "Canceled"]
    )
    plt.title(f"Confusion Matrix With CV - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"cm_cv_{model_name.replace(' ', '_')}.png", dpi=300)
    plt.show()

"""# 18. Feature Importance"""

# Feature Importance for Random Forest model

# Random Forest without CV
rf_importance_no_cv = model_rf.featureImportances if hasattr(model_rf, "featureImportances") else None

# Random Forest with CV is inside the PipelineModel/CrossValidatorModel
try:
    rf_cv_model = model_rf.stages[0].bestModel
    rf_importance_cv = rf_cv_model.featureImportances
except Exception:
    rf_importance_cv = None

# Use CV feature importance if available; otherwise use no CV
importance_values = rf_importance_cv if rf_importance_cv is not None else rf_importance_no_cv

feature_importance_df = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": importance_values.toArray()
}).sort_values(by="Importance", ascending=False)

print(feature_importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(
    data=feature_importance_df,
    x="Importance",
    y="Feature"
)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()