# Baguette pipeline
Seriously Rubios Baguettes, Big Data Analytics, University of Warsaw, winter semester 2026-2027.

This project uses two years of JustJoinIt job offers (Kaggle dataset, October 2021 to September 2023, about 9GB) to find out when, where and in which roles data students can realistically get a junior job. It ends with a recommendation like: "Junior offers peak in month X, role Y has N times more junior openings than Data Scientist, so apply to Y in X."

## Questions
- Which data roles have the most junior offers, and what share of each role's offers is junior.
- Where the junior offers are: city, remote or hybrid.
- Which contract types juniors get (B2B or employment contract), and at what salary.
- Which companies hire juniors most often.
- Which months junior offers peak in.
- Which adjacent roles have a lower entry bar than Data Scientist.

## Stack
Databricks Free Edition, Python and PySpark, and Delta tables in a medallion architecture (ELT):

- **Bronze:** the raw records, loaded once and left unchanged, plus the ingestion timestamp and source file name.
- **Silver:** one row per unique offer, with one schema across all periods. Reposted offers are deduplicated, multi-location offers are handled, seniority and role are normalised, and salaries are converted to PLN per month by contract type.
- **Gold:** one aggregate table per question.

The raw dataset is not included in this repo. Download it from Kaggle.

## Deliverables
- A 2-page paper on the methodology.
- A 5-10 minute presentation on 30.11.2026.
- Databricks notebooks with the pipeline and the analysis.
