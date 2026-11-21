# Databricks notebook source
# MAGIC %md
# MAGIC # Bronze: raw JustJoinIt offers
# MAGIC This notebook loads all 671 daily snapshot files as they are into `seriously_rubios_baguettes.bronze.offers`. It adds three columns: `snapshot_date`, `source_file` and `ingested_at`. It does no cleaning, filtering or dedup; all of that happens in silver.
# MAGIC
# MAGIC Steps:
# MAGIC 1. Unzip the Kaggle zip from the `raw` volume into one JSON file per day. Files that already exist are skipped, so reruns are cheap.
# MAGIC 2. Read every file with Spark, which takes the union of all fields across periods (fields added later are null for earlier days).
# MAGIC 3. Overwrite the bronze table, then check counts.
# MAGIC
# MAGIC Safe to rerun. It rebuilds bronze from the raw files every time.

# COMMAND ----------

import os
import shutil
import zipfile

from pyspark.sql import functions as F

VOLUME = "/Volumes/seriously_rubios_baguettes/bronze/raw"
ZIP_PATH = f"{VOLUME}/zip/justjoinit-job-offers-data-2021-10-2023-09.zip"
JSON_DIR = f"{VOLUME}/json"
TABLE = "seriously_rubios_baguettes.bronze.offers"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Unzip into daily JSON files

# COMMAND ----------

written, skipped = 0, 0
with zipfile.ZipFile(ZIP_PATH) as z:
    members = [m for m in z.infolist() if m.filename.endswith(".json")]
    for m in members:
        target = os.path.join(JSON_DIR, m.filename)  # e.g. json/2021-10/2021-10-23.json
        if os.path.exists(target) and os.path.getsize(target) == m.file_size:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with z.open(m) as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst, length=16 * 1024 * 1024)
        written += 1

print(f"{len(members)} files in zip: {written} written, {skipped} already there")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Read all daily files
# MAGIC Each file is a single JSON array, so `multiLine` is needed. `snapshot_date` comes from the file name, not from the records.

# COMMAND ----------

raw = (
    spark.read.option("multiLine", True)
    .json(f"{JSON_DIR}/*/*.json")
    .withColumn("source_file", F.col("_metadata.file_path"))
    .withColumn("snapshot_date", F.to_date(F.regexp_extract("source_file", r"(\d{4}-\d{2}-\d{2})\.json$", 1)))
    .withColumn("ingested_at", F.current_timestamp())
)
raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write bronze

# COMMAND ----------

(
    raw.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(TABLE)
)
spark.sql(f"COMMENT ON TABLE {TABLE} IS 'Raw JustJoinIt daily snapshots (Kaggle, 2021-10 to 2023-09), one row per record per snapshot day. No cleaning.'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Checks
# MAGIC Expect 671 snapshot days, no null `snapshot_date`, and roughly 6-18k rows per day.

# COMMAND ----------

bronze = spark.table(TABLE)
checks = bronze.agg(
    F.count("*").alias("rows"),
    F.countDistinct("snapshot_date").alias("snapshot_days"),
    F.countDistinct("source_file").alias("source_files"),
    F.sum(F.col("snapshot_date").isNull().cast("int")).alias("null_snapshot_date"),
    F.min("snapshot_date").alias("first_day"),
    F.max("snapshot_date").alias("last_day"),
)
display(checks)

# COMMAND ----------

display(
    bronze.groupBy(F.date_format("snapshot_date", "yyyy-MM").alias("month"))
    .agg(F.countDistinct("snapshot_date").alias("days"), F.count("*").alias("rows"))
    .orderBy("month")
)

# COMMAND ----------

# Summary for runs started from the CLI or a job
import json
dbutils.notebook.exit(json.dumps({**checks.first().asDict(), "columns": len(bronze.columns)}, default=str))
