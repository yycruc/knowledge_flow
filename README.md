# Machine learning citations in LIS

Code and analysis data for the article:

> How is Machine Learning research cited in Library and Information Science
> literature?

## Files

```text
scim_analysis/
  README.md
  requirements.txt
  analysis.py
  data/
    annual_metrics.csv
    citation_lag_metrics.csv
    paper_ml_share.csv
    task_metrics.csv
    boundary_checks.csv
    journal_classification.csv
    arxiv_coverage.csv
    topic_search_terms.csv
```

The script creates the seven article figures, the three main tables, and the
Appendix B-C tables in `results/`.

## Run

```bash
python -m pip install -r requirements.txt
python analysis.py
```

## Data

- `annual_metrics.csv`: annual values for Figures 1-4.
- `citation_lag_metrics.csv`: values for Tables 1-2 and Figure 5.
- `paper_ml_share.csv`: anonymized article-level values for Figure 6.
- `task_metrics.csv`: fractional task shares for Table 3 and Figure 7.
- `boundary_checks.csv`: alternative journal-corpus specifications for
  Appendix B, Tables B3-B5.
- `journal_classification.csv`: journal groups and corpus counts for Appendix
  B, Table B2.
- `arxiv_coverage.csv`: selected arXiv-category coverage values for Appendix
  B, Table B1.
- `topic_search_terms.csv`: search terms listed in Appendix C.
