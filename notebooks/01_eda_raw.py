# Databricks notebook source
# MAGIC %md
# MAGIC # EDA: raw JustJoinIt daily files
# MAGIC This notebook explores a few raw daily snapshot files before silver is built. It reads them straight from the volume `seriously_rubios_baguettes.bronze.raw` (`json/YYYY-MM/YYYY-MM-DD.json`) with pandas. All 671 days are there.
# MAGIC
# MAGIC **How to use:** set the days you want in `DAYS` (section 2) and run everything top to bottom. Five days take a few seconds. Don't load more than about 20 days with pandas.
# MAGIC
# MAGIC Note on `display_offer`: from 2022 onwards, a multi-city offer is stored once per city, and only the parent record has `display_offer = True`. The 2021 files don't have the field, so their records are treated as displayed. Most counts below use only the displayed records (`dfd`).

# COMMAND ----------

import json
import os
import re

import pandas as pd

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)

JSON_DIR = "/Volumes/seriously_rubios_baguettes/bronze/raw/json"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Files in the volume
# MAGIC This shows one file per day, the size of each month, and which days are missing.

# COMMAND ----------

rows = []
for month in sorted(os.listdir(JSON_DIR)):
    for name in sorted(os.listdir(f"{JSON_DIR}/{month}")):
        rows.append({"month": month, "day": name[:10], "mb": os.path.getsize(f"{JSON_DIR}/{month}/{name}") / 1e6})
files = pd.DataFrame(rows)
files["day"] = pd.to_datetime(files["day"])
print(f"{len(files)} files, {files.mb.sum() / 1000:.1f} GB")
files.groupby("month").agg(files=("day", "size"), mb=("mb", "sum"), avg_mb_per_day=("mb", "mean")).round(1)

# COMMAND ----------

all_days = pd.date_range(files.day.min(), files.day.max(), freq="D")
missing = all_days.difference(files.day)
print(f"{len(missing)} missing days between {files.day.min():%Y-%m-%d} and {files.day.max():%Y-%m-%d}")
pd.Series(missing.strftime("%Y-%m-%d")).groupby(missing.strftime("%Y-%m")).agg(list)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load the chosen days
# MAGIC There's one row per record per snapshot day, kept exactly as in the file, plus `snapshot_date` taken from the file name.

# COMMAND ----------

DAYS = ["2021-10-23", "2021-10-24", "2022-09-01", "2023-03-01", "2023-09-25"]
# e.g. the 1st of every quarter: [d.strftime("%Y-%m-%d") for d in files.day if d.day == 1 and d.month in (1, 4, 7, 10)]

raw = {}  # snapshot_date -> list of raw records, as in the file
for d in DAYS:
    with open(f"{JSON_DIR}/{d[:7]}/{d}.json", encoding="utf-8") as f:
        raw[d] = json.load(f)

df = pd.concat([pd.DataFrame(recs).assign(snapshot_date=d) for d, recs in raw.items()], ignore_index=True)
df["is_displayed"] = df["display_offer"].map(lambda v: v is not False) if "display_offer" in df else True
dfd = df[df.is_displayed]  # displayed offers: one per multi-city group
days = sorted(raw)
print(f"{len(df):,} records from {len(days)} days: {days}")
df.groupby("snapshot_date").agg(records=("id", "size"), displayed=("is_displayed", "sum"), unique_ids=("id", "nunique"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Schema
# MAGIC This shows which fields exist on which day. The 2021 files have 20 fields; `multilocation`, `display_offer`, `open_to_hire_ukrainians` and `way_of_apply` appear later.

# COMMAND ----------

fields = {d: set().union(*(r.keys() for r in recs)) for d, recs in raw.items()}
all_fields = sorted(set().union(*fields.values()))
pd.DataFrame({d: ["✓" if f in fs else "" for f in all_fields] for d, fs in fields.items()}, index=all_fields)

# COMMAND ----------

# One full raw record: change the day or the row to look at others
day, row = days[-1], 0
print(json.dumps(raw[day][row], ensure_ascii=False, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Each file is a snapshot of every offer listed that day
# MAGIC This is the share of ids from the row day that are still present on the column day. Neighbouring days overlap by about 97%, so the pipeline has to dedup across days.

# COMMAND ----------

ids = {d: {r["id"] for r in recs} for d, recs in raw.items()}
pd.DataFrame({b: {a: len(ids[a] & ids[b]) / len(ids[a]) for a in days} for b in days}).round(3)

# COMMAND ----------

# How long had each offer been listed on the first day? published_at compared with snapshot_date
first = df[df.snapshot_date == days[0]].copy()
published = pd.to_datetime(first.published_at, utc=True).dt.tz_localize(None).dt.normalize()
first["age_days"] = (pd.to_datetime(first.snapshot_date) - published).dt.days
first.age_days.describe(percentiles=[.25, .5, .75, .9, .99]).round(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Multi-location offers
# MAGIC From 2022 onwards, one parent record lists every city in `multilocation`, and each city also gets a child record (`display_offer = False`, id with a city suffix).

# COMMAND ----------

has_ml = df[df["multilocation"].notna()].copy()
has_ml["n_cities"] = has_ml["multilocation"].map(len)
pd.crosstab([has_ml.snapshot_date, has_ml.is_displayed], has_ml.n_cities.clip(upper=10).rename("n_cities (10 = 10+)"))

# COMMAND ----------

# One parent offer and its child records on the same day
day = days[-1]
parent = has_ml[(has_ml.snapshot_date == day) & (has_ml.n_cities > 1)].iloc[0]
child_slugs = {m["slug"] for m in parent["multilocation"]}
print("Parent:", parent["id"], "| cities:", [m["city"] for m in parent["multilocation"]])
df[(df.snapshot_date == day) & df["id"].isin(child_slugs)][["id", "city", "display_offer", "published_at", "experience_level", "title"]]

# COMMAND ----------

# 2021 has no multilocation field, but some title + company pairs still repeat within a day
day = days[0]
d21 = df[df.snapshot_date == day]
dupes = d21[d21.duplicated(["company_name", "title"], keep=False)]
print(f"{day}: {len(d21):,} records, {d21.groupby(['company_name', 'title']).ngroups:,} distinct company + title pairs, {len(dupes):,} records in repeated pairs")
dupes.sort_values(["company_name", "title"])[["company_name", "title", "id", "city", "published_at", "experience_level"]].head(40)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Seniority, and intern or trainee offers
# MAGIC `experience_level` is only junior, mid or senior. Intern and trainee offers can only be found by matching keywords in the title.

# COMMAND ----------

lvl = pd.crosstab(dfd.snapshot_date, dfd.experience_level)
lvl.join((lvl.div(lvl.sum(axis=1), axis=0) * 100).round(1).add_suffix(" %"))

# COMMAND ----------

INTERN_RE = re.compile(r"intern|trainee|sta[zż]|praktyk|apprentice|graduate|absolwent", re.I)
interns = dfd[dfd.title.str.contains(INTERN_RE)]
pd.crosstab(interns.snapshot_date, interns.experience_level)

# COMMAND ----------

interns[["snapshot_date", "experience_level", "marker_icon", "title", "company_name"]].head(40)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Categories and data roles
# MAGIC `marker_icon` is the category (24 values). There's no role field, so roles have to come from the title.

# COMMAND ----------

pd.crosstab(dfd.marker_icon, dfd.snapshot_date).sort_values(days[-1], ascending=False)

# COMMAND ----------

# Data-like titles by category: what would only using 'data' and 'analytics' miss?
DATA_RE = re.compile(r"\bdata\b|danych|analy|analit|\bBI\b|business intelligence|machine learning|\bML\b|\bAI\b|\bETL\b|power bi|tableau", re.I)
data_like = dfd[dfd.title.str.contains(DATA_RE)]
pd.crosstab(data_like.marker_icon, data_like.snapshot_date).sort_values(days[-1], ascending=False)

# COMMAND ----------

# Data-like titles outside data/analytics: are they data roles?
outside = data_like[~data_like.marker_icon.isin(["data", "analytics"])]
outside[["snapshot_date", "marker_icon", "experience_level", "title"]].sample(min(40, len(outside)), random_state=1)

# COMMAND ----------

# DRAFT role rules to experiment with: change the patterns and see what moves
ROLE_RULES = [
    ("ML Engineer",    r"machine learning engineer|\bML engineer|\bMLOps\b"),
    ("Data Scientist", r"data scien|machine learning|\bML\b|\bAI\b"),
    ("Data Engineer",  r"data engineer|big data|\bETL\b|inżynier danych|data platform"),
    ("BI",             r"\bBI\b|business intelligence|power bi|tableau|reporting"),
    ("Data Analyst",   r"data analy|analityk danych|analiz|\banalyst\b|analityk"),
]
def guess_role(title: str) -> str:
    for role, pattern in ROLE_RULES:
        if re.search(pattern, title, re.I):
            return role
    return "Other"

data_cat = dfd[dfd.marker_icon.isin(["data", "analytics"])].copy()
data_cat["role"] = data_cat.title.map(guess_role)
pd.crosstab([data_cat.snapshot_date, data_cat.role], data_cat.experience_level, margins=True)

# COMMAND ----------

# Junior titles in data/analytics: what the 'Other' bucket is hiding
data_cat[data_cat.experience_level == "junior"][["snapshot_date", "marker_icon", "role", "title", "company_name"]]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Salaries
# MAGIC There's one row per contract type. B2B amounts are net invoice and `permanent` (umowa o pracę) amounts are gross, so don't compare them directly.

# COMMAND ----------

sal = dfd[["snapshot_date", "id", "experience_level", "marker_icon", "employment_types"]].explode("employment_types").dropna(subset=["employment_types"])
sal["type"] = sal.employment_types.str["type"]
s = sal.employment_types.str["salary"]
sal["has_salary"] = s.notna()
sal["from"], sal["to"], sal["currency"] = s.str["from"], s.str["to"], s.str["currency"]
sal = sal.drop(columns="employment_types")

sal.groupby(["snapshot_date", "type"]).has_salary.agg(entries="size", share_with_salary="mean").round(2)

# COMMAND ----------

sal[sal.has_salary].groupby("snapshot_date").currency.value_counts().unstack(fill_value=0)

# COMMAND ----------

pln = sal[sal.has_salary & (sal.currency == "pln")]
pln.groupby(["type", "experience_level"])[["from", "to"]].describe(percentiles=[.1, .5, .9]).round(0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Location and work mode

# COMMAND ----------

wt = pd.crosstab(dfd.snapshot_date, dfd.workplace_type)
wt.join((wt.div(wt.sum(axis=1), axis=0) * 100).round(1).add_suffix(" %"))

# COMMAND ----------

pd.crosstab(dfd.city, dfd.snapshot_date).sort_values(days[-1], ascending=False).head(20)

# COMMAND ----------

dfd.country_code.value_counts(dropna=False).head(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Skills in junior data offers

# COMMAND ----------

jd = data_cat[data_cat.experience_level == "junior"].explode("skills").dropna(subset=["skills"])
jd["skill"] = jd.skills.str["name"].str.strip()
jd.skill.value_counts().head(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Playground
# MAGIC Useful frames:
# MAGIC - `df`: every record.
# MAGIC - `dfd`: displayed offers only.
# MAGIC - `data_cat`: data and analytics offers, with a guessed `role`.
# MAGIC - `sal`: one row per salary entry.
# MAGIC - `files`: the file list.
# MAGIC - `raw`: the raw records for each day.
# MAGIC
# MAGIC For the full dataset, query the Spark table `seriously_rubios_baguettes.bronze.offers` instead.

# COMMAND ----------

dfd[dfd.title.str.contains("junior", case=False)].marker_icon.value_counts()
