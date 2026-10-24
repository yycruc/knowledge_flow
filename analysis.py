"""Recompute the final article results from the bundled, frozen source records.

Units and windows are intentionally explicit: citation links are unique pairs,
task weights are per original annotation, and reference ages are first averaged
within each citing publication. No historical output is read by this module.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from functools import lru_cache
from io import StringIO
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import stats

YEARS = range(2017, 2023)
RESIDUAL = {'Methodology', 'Miscellaneous'}
ARXIV_CATEGORIES = ['stat.ML', 'cs.CV', 'cs.LG', 'cs.CL', 'cs.NE', 'cs.AI']
MODELS = ['Full corpus', 'No IPM', 'No CS/IR boundary',
          'No CS/IR + MIS/IS boundary', 'Core LIS/IS/scientometrics only',
          'No ASJC-CS stress test']


def normalize_doi(value):
    if pd.isna(value):
        return ''
    value = str(value).strip().lower()
    value = re.sub(r'^(?:https?://(?:dx\.)?doi\.org/|doi\s*:\s*)', '', value).rstrip(' .;,')
    return value if re.match(r'^10\.\d{4,9}/\S+$', value) and not re.search(r'n{4,}|x{5,}', value) else ''


def normalize_arxiv(value):
    if pd.isna(value):
        return ''
    value = str(value).strip().lower()
    value = re.sub(r'^.*(?:arxiv.org/abs/|arxiv:)', '', value)
    value = re.sub(r'v\d+$', '', value)
    match = re.fullmatch(r'(\d{4})\.(\d+)', value)
    if match:
        width = 4 if int(match[1]) < 1501 else 5
        return match[1] + '.' + match[2].ljust(width, '0') if len(match[2]) <= width else ''
    return value if re.fullmatch(r'[a-z.-]+/\d{7}', value) else ''


def parse_labels(value):
    if pd.isna(value):
        return []
    try:
        labels = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        labels = json.loads(value)
    if isinstance(labels, str):
        labels = [labels]
    return list(dict.fromkeys(str(label).strip() for label in labels if str(label).strip()))


def hierarchy_mapper(table, kind):
    """Retain archived path multiplicities, including repeated destinations."""
    records = table.fillna('').values.tolist()
    roots, seconds = set(table.iloc[:, 0].dropna()), set(table.iloc[:, 1].dropna())

    @lru_cache(None)
    def allocate(token, level):
        if kind == 'task' and token == 'Named Entity Recognition (NER)':
            token = 'Named Entity Recognition'
        if (level == 1 and token in roots) or (level == 2 and token in seconds):
            return {token: 1.0}
        paths = []
        for root, parent, *children in records:
            if level == 1:
                if token == parent:
                    paths.append(root)
                else:
                    paths.extend(root for child in children if token == child)
            else:
                paths.extend(parent for child in children if token == child)
        if not paths:
            return {token: 1.0}
        return {destination: count / len(paths) for destination, count in Counter(paths).items()}

    return allocate, roots, seconds


def numeric_csv_boundary(frame):
    """Use one consistent CSV numeric parsing convention at aggregation boundaries.

    This mirrors file-backed calculations without retaining redundant intermediate
    files. Exact IEEE-754 distinctions matter for tied growth ranks; do not round
    weights or growth rates before a correlation test.
    """
    return pd.read_csv(StringIO(frame.to_csv(index=False)))


def consolidate_pwc(data, summary):
    p = pd.read_csv(data / 'pwc_publications.csv.gz', dtype=str).fillna('')
    p['source_row'] = np.arange(len(p))
    p['ax'] = p.arxiv_id.map(normalize_arxiv)
    p['doi_key'] = p.doi.map(normalize_doi)
    alias = pd.read_csv(data / 'pwc_identifier_aliases.csv.gz', dtype=str).fillna('')
    aliases = alias.groupby('arxiv_id').doi.agg(lambda values: sorted(set(values)))
    p['alias_keys'] = p.ax.map(aliases).map(lambda value: value if isinstance(value, list) else [])
    valid = p[p.doi_key.ne('')]
    distinct = valid.groupby('doi_key').ax.agg(lambda values: len(set(values) - {''}))
    blocked = set(distinct[distinct.gt(1)].index)
    unique_ax = valid.groupby('doi_key').ax.agg(
        lambda values: next(iter(set(values) - {''})) if len(set(values) - {''}) == 1 else '')
    p['publication_key'] = [
        'ax:' + ax if ax else ('ax:' + unique_ax.get(doi, '') if unique_ax.get(doi, '')
                              else 'doi:' + doi if doi else 'row:' + str(row))
        for ax, doi, row in zip(p.ax, p.doi_key, p.source_row)]
    p['task_list'] = p.tasks.map(parse_labels)
    p['method_list'] = p.methods.map(parse_labels)
    p['year'] = pd.to_numeric(p.date.str[:4], errors='coerce')
    groups = {}
    for row in p.itertuples():
        group = groups.setdefault(row.publication_key, {
            'publication_key': row.publication_key, 'arxiv_id': row.ax,
            'years': Counter(), 'tasks': {}, 'methods': {}})
        if row.ax:
            group['arxiv_id'] = row.ax
        if pd.notna(row.year):
            group['years'][int(row.year)] += 1
        group['tasks'].update(dict.fromkeys(row.task_list))
        group['methods'].update(dict.fromkeys(row.method_list))
    canonical = []
    for group in groups.values():
        years = group.pop('years')
        # Modal year; earliest year breaks a tie. Preserve annotation order.
        group['year'] = sorted(years.items(), key=lambda item: (-item[1], item[0]))[0][0] if years else None
        group['tasks'], group['methods'] = list(group['tasks']), list(group['methods'])
        canonical.append(group)
    canonical = pd.DataFrame(canonical)
    by_doi, by_arxiv = defaultdict(set), defaultdict(set)
    for row in p.itertuples():
        if row.doi_key and row.doi_key not in blocked:
            by_doi[row.doi_key].add(row.publication_key)
        for key in row.alias_keys:
            if key and key not in blocked:
                by_doi[key].add(row.publication_key)
        if row.ax:
            by_arxiv[row.ax].add(row.publication_key)
    blocked |= {key for key, values in by_doi.items() if len(values) > 1}
    by_doi = {key: next(iter(values)) for key, values in by_doi.items() if len(values) == 1 and key not in blocked}
    by_arxiv = {key: next(iter(values)) for key, values in by_arxiv.items() if len(values) == 1}
    summary['pwc'] = {'input_records': len(p), 'canonical_publications': len(canonical),
                      'withheld_ambiguous_dois': len(blocked),
                      'invalid_nonempty_dois': int(p.doi.ne('').sum() - p.doi_key.ne('').sum())}
    return canonical, by_doi, by_arxiv


def match_references(data, canonical, by_doi, by_arxiv, summary):
    w = pd.read_csv(data / 'lis_references.csv.gz', dtype={
        'citing_doi': str, 'cited_doi': str, 'cited_arxiv_id': str}, low_memory=False)
    w = w.rename(columns={'citing_doi': 'doi', 'cited_doi': 'rdoi', 'cited_arxiv_id': 'arxiv_id',
                          'document_type': 'type', 'citing_year': 'year', 'cited_year': 'ryear',
                          'fields_of_study': 'field'})
    w['source_reference_row'] = np.arange(len(w))
    raw_count = len(w)
    w['doi'], w['rdoi'], w['ax'] = w.doi.map(normalize_doi), w.rdoi.map(normalize_doi), w.arxiv_id.map(normalize_arxiv)
    w = w[w.year.between(2017, 2022) & w.doi.ne('')].copy()
    w['lag'] = w.year - w.ryear
    missing, negative = int(w.lag.isna().sum()), int(w.lag.lt(0).sum())
    w = w[w.lag.ge(0)].copy()
    w['reference_key'] = ['doi:' + doi if doi else 'ax:' + ax if ax else 'unresolved-row:' + str(row)
                          for doi, ax, row in zip(w.rdoi, w.ax, w.source_reference_row)]
    if w.groupby('doi').year.nunique().gt(1).any():
        raise ValueError('A citing DOI has conflicting publication years')
    if w.groupby(['doi', 'reference_key']).ryear.nunique().gt(1).any():
        raise ValueError('Duplicate reference identities have conflicting cited years')
    duplicates = int(w.duplicated(['doi', 'reference_key']).sum())
    w = w.drop_duplicates(['doi', 'reference_key']).copy()
    w['journal'] = w.journal.fillna('').str.lower()
    metadata = canonical.set_index('publication_key').to_dict('index')
    keys, links, conflicts = [], [], 0
    for row in w.itertuples():
        doi_key, ax_key = by_doi.get(row.rdoi), by_arxiv.get(row.ax)
        if doi_key and ax_key and doi_key != ax_key:
            key = None
            conflicts += 1
        else:
            key = doi_key or ax_key
        keys.append(key)
        if key:
            links.append({'wos_doi': row.doi, 'publication_key': key,
                          'journal': row.journal, 'year': int(row.year),
                          'ryear': int(row.ryear), 'lag': row.lag,
                          'tasks': metadata[key]['tasks'], 'methods': metadata[key]['methods']})
    w['publication_key'], w['is_ml'] = keys, pd.notna(keys)
    links = pd.DataFrame(links)
    assert not links.duplicated(['wos_doi', 'publication_key']).any()
    assert len(links) == w.is_ml.sum()
    summary['corpus'] = {'input_reference_rows': raw_count, 'references': len(w),
                          'citing_publications': int(w.doi.nunique()), 'journals': int(w.journal.nunique()),
                          'ml_links': len(links), 'ml_citing_publications': int(links.wos_doi.nunique()),
                          'cited_pwc_publications': int(links.publication_key.nunique()),
                          'ml_journals': int(links.journal.nunique()),
                          'excluded_missing_year': missing, 'excluded_negative_age': negative,
                          'removed_duplicate_rows': duplicates, 'conflicting_doi_arxiv_matches': conflicts}
    return w, links


def expand_entities(links, canonical, data, summary):
    expanded, baseline, coverage, catalogues = {}, {}, [], {}
    for kind, column in [('task', 'tasks'), ('method', 'methods')]:
        hierarchy = pd.read_csv(data / f'{kind}_hierarchy.csv.gz')
        allocate, roots, seconds = hierarchy_mapper(hierarchy, kind)
        catalogues[kind] = (hierarchy, roots, seconds)
        for level, catalogue in [(1, roots), (2, seconds)]:
            records, pwc_counts = [], Counter()
            for row in links.itertuples():
                for token in getattr(row, column):
                    for destination, weight in allocate(token, level).items():
                        records.append({'task': destination, 'raw_label': token, 'year': row.year,
                                        'journal': row.journal, 'count': weight,
                                        'mapped': destination in catalogue, 'wos_doi': row.wos_doi,
                                        'publication_key': row.publication_key})
            frame = numeric_csv_boundary(pd.DataFrame(records))
            expanded[(kind, level)] = frame
            assert np.isclose(frame['count'].sum(), sum(len(labels) for labels in links[column]), atol=1e-7)
            if kind == 'task':
                for row in canonical[canonical.year.between(2010, 2022)].itertuples():
                    for token in row.tasks:
                        for destination, weight in allocate(token, level).items():
                            pwc_counts[(destination, int(row.year))] += weight
                baseline[level] = numeric_csv_boundary(pd.DataFrame([
                    {'task': key[0], 'year': key[1], 'count': value, 'mapped': key[0] in catalogue}
                    for key, value in pwc_counts.items()]))
            for year, group in frame.groupby('year'):
                n = int(group.loc[group.mapped, 'task'].nunique())
                coverage.append({'kind': kind, 'level': level, 'year': year,
                                 'observed_catalog_entities': n, 'catalog_entities': len(catalogue),
                                 'coverage_pct': n / len(catalogue) * 100})
    summary['catalogues'] = {kind: {'first_level': len(item[1]), 'second_level': len(item[2])}
                             for kind, item in catalogues.items()}
    return expanded, baseline, pd.DataFrame(coverage), catalogues


def compute(data: Path, output: Path):
    tables_dir, figures_data = output / 'tables', output / 'figure_data'
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_data.mkdir(parents=True, exist_ok=True)
    annotations = json.loads((data / 'study_annotations.json').read_text(encoding='utf8'))
    summary = {'windows': {'lis': [2017, 2022], 'table4_pwc_baseline': [2010, 2022],
                            'figure7_and_growth': [2017, 2022]}}
    print('1/6 Consolidating PwC publications and identifier aliases', flush=True)
    canonical, by_doi, by_arxiv = consolidate_pwc(data, summary)
    print('2/6 Filtering, deduplicating and matching LIS references', flush=True)
    w, links = match_references(data, canonical, by_doi, by_arxiv, summary)
    summary['corpus']['source_journals'] = annotations['source_journals']

    print('3/6 Computing publication trends and citation ages', flush=True)
    topic = pd.read_csv(data / 'topic_search.csv.gz', dtype={'doi': str})
    topic['doi'] = topic.doi.map(normalize_doi)
    topic = topic[topic.year.between(2017, 2022) & topic.doi.isin(set(w.doi))].drop_duplicates('doi').copy()
    topic['year'] = topic.doi.map(w.groupby('doi').year.first()).astype(int)
    citation_set, topic_set = set(links.wos_doi), set(topic.doi)
    common = len(citation_set & topic_set)
    summary['overlap'] = {'citation_set': len(citation_set), 'topic_set': len(topic_set),
                           'intersection': common, 'citation_pct': common / len(citation_set) * 100,
                           'topic_pct': common / len(topic_set) * 100}
    annual = []
    for year in YEARS:
        references, matched = w[w.year.eq(year)], links[links.year.eq(year)]
        papers, citing = references.doi.nunique(), matched.wos_doi.nunique()
        topic_count = topic[topic.year.eq(year)].doi.nunique()
        annual.append({'year': year, 'lis_publications': papers, 'references': len(references),
                       'ml_citing_publications': citing, 'ml_links': len(matched), 'topic_publications': topic_count,
                       'citation_paper_share': citing / papers * 100, 'topic_paper_share': topic_count / papers * 100,
                       'ml_reference_share': len(matched) / len(references) * 100})
    annual = pd.DataFrame(annual)
    ml_age, all_age = links.groupby('wos_doi').lag.mean(), w.groupby('doi').lag.mean()
    age_table = pd.DataFrame([{'group': group, 'n': len(values), 'mean': values.mean(),
                               'median': values.median(), 'std': values.std(ddof=1)}
                              for group, values in [('ML publications', ml_age), ('All citations', all_age)]])
    annual_age = []
    for year, group in links.groupby('year'):
        values = group.groupby('wos_doi').lag.mean()
        annual_age.append({'year': year, 'n': len(values), 'mean': values.mean(),
                           'median': values.median(), 'std': values.std(ddof=1)})
    w['fields'] = w.field.map(parse_labels)
    fields = w[~w.is_ml].explode('fields').groupby(['fields', 'doi']).lag.mean().reset_index()
    fields = fields.groupby('fields').lag.agg(['mean', 'median', 'count']).reset_index().rename(columns={'fields': 'field'})
    all_n, ml_n = w.groupby('doi').size(), links.groupby('wos_doi').size()
    nonml_age = w[~w.is_ml].groupby('doi').lag.mean()
    figure6 = pd.DataFrame({'ml_share': ml_n / all_n, 'nonml_mean_age': nonml_age,
                           'ml_mean_age': ml_age, 'all_reference_count': all_n, 'ml_reference_count': ml_n})
    figure6 = figure6.reindex(ml_age.index).reset_index().rename(columns={'index': 'doi', 'wos_doi': 'doi'})
    figure6 = figure6.dropna(subset=['ml_share', 'nonml_mean_age'])
    test = stats.pearsonr(figure6.ml_share, figure6.nonml_mean_age)
    ci = test.confidence_interval(confidence_level=0.95)
    statistics6 = pd.DataFrame([{'n': len(figure6), 'r': test.statistic, 'p': test.pvalue,
                                 'ci_low': ci.low, 'ci_high': ci.high}])
    summary['figure6'] = statistics6.iloc[0].to_dict()
    summary['ipm'] = {'ml_links': int(links.journal.eq('information processing & management').sum()),
                       'ml_link_share_pct': links.journal.eq('information processing & management').mean() * 100}

    print('4/6 Allocating task and method annotations through the archived hierarchy', flush=True)
    expanded, baseline, coverage, catalogues = expand_entities(links, canonical, data, summary)
    first, second = expanded[('task', 1)], expanded[('task', 2)]
    lis_counts = first.groupby('task')['count'].sum()
    pwc_counts = baseline[1].groupby('task')['count'].sum()
    task_shares = pd.DataFrame({'lis_count': lis_counts, 'pwc_count': pwc_counts})
    task_shares['lis_share'] = task_shares.lis_count / lis_counts.sum() * 100
    task_shares['pwc_share'] = task_shares.pwc_count / pwc_counts.sum() * 100
    task_shares['mapped'] = task_shares.index.isin(catalogues['task'][1])
    table4 = task_shares[task_shares.mapped & ~task_shares.index.isin(RESIDUAL)].sort_values('lis_share', ascending=False).head(10)
    table4 = table4.reset_index()
    table4.insert(0, 'rank', range(1, len(table4) + 1))
    comparable = pd.DataFrame({'lis_count': lis_counts,
                               'pwc_count': baseline[1][baseline[1].year.between(2017, 2022)].groupby('task')['count'].sum()}).dropna()
    comparable = comparable[(comparable.lis_count > 0) & (comparable.pwc_count > 0) & ~comparable.index.isin(RESIDUAL)].copy()
    comparable['hierarchy_status'] = np.where(comparable.index.isin(catalogues['task'][1]), 'First-level category', 'Unmapped task label')
    comparable['lis_share_pct'] = comparable.lis_count / comparable.lis_count.sum() * 100
    comparable['pwc_share_pct'] = comparable.pwc_count / comparable.pwc_count.sum() * 100
    comparable['selectivity_log2'] = np.log2(comparable.lis_share_pct / comparable.pwc_share_pct)
    comparable['direction'] = np.select([comparable.selectivity_log2 > .25, comparable.selectivity_log2 < -.25],
                                         ['Over-represented', 'Under-represented'], default='Near proportional')
    figure7 = comparable.sort_values('lis_share_pct', ascending=False).reset_index()
    assert np.isclose(figure7.lis_share_pct.sum(), 100) and np.isclose(figure7.pwc_share_pct.sum(), 100)
    summary['figure7'] = {'points': len(figure7), 'mapped_first_level': int(figure7.hierarchy_status.eq('First-level category').sum()),
                           'lis_denominator': comparable.lis_count.sum(), 'pwc_denominator': comparable.pwc_count.sum()}

    print('5/6 Computing all six journal specifications and 24 correlation tests', flush=True)
    journal_metadata = pd.DataFrame(annotations['journals'])
    j = journal_metadata[journal_metadata.journal.isin(set(w.journal))].copy()
    assert set(j.journal) == set(w.journal)
    assert j.journal.is_unique
    # Complete codes, including every secondary code, determine M5 directly.
    j['exclude_asjc_cs'] = j.asjc_codes.map(lambda codes: any(code.startswith('17') for code in codes))
    masks = [pd.Series(True, index=j.index), j.journal.ne('information processing & management'),
             ~j.exclude_cs_ir, ~(j.exclude_cs_ir | j.exclude_mis_is),
             j['group'].str.startswith('Core LIS'), ~(j.exclude_cs_ir | j.exclude_asjc_cs)]
    boundary_summary, boundary_annual, correlations, growth_details = [], [], [], []
    for model, mask in enumerate(masks):
        names = set(j.loc[mask, 'journal'])
        reference_subset, link_subset = w[w.journal.isin(names)], links[links.journal.isin(names)]
        first_subset, second_subset = first[first.journal.isin(names)], second[second.journal.isin(names)]
        top = first_subset[first_subset.mapped].groupby('task')['count'].sum().sort_values(ascending=False)
        boundary_summary.append({'model': f'M{model}', 'specification': MODELS[model], 'journals': len(names),
                                 'lis_publications': reference_subset.doi.nunique(), 'references': len(reference_subset),
                                 'ml_links': len(link_subset), 'ml_share': len(link_subset) / len(reference_subset) * 100,
                                 'top_first_task': top.index[0], 'top_share': top.iloc[0] / first_subset['count'].sum() * 100})
        for year in YEARS:
            ref_year, link_year = reference_subset[reference_subset.year.eq(year)], link_subset[link_subset.year.eq(year)]
            boundary_annual.append({'model': f'M{model}', 'year': year, 'references': len(ref_year),
                                    'ml_links': len(link_year), 'ml_share': len(link_year) / len(ref_year) * 100})
        for level, lis in [(1, first_subset), (2, second_subset)]:
            lp = lis[lis.mapped].pivot_table(index='task', columns='year', values='count', aggfunc='sum', fill_value=0)
            pwc = baseline[level]
            pp = pwc[pwc.mapped].pivot_table(index='task', columns='year', values='count', aggfunc='sum', fill_value=0)
            growth = lp[[2017, 2022]].join(pp[[2017, 2022]], how='inner', lsuffix='_lis', rsuffix='_pwc')
            if level == 1:
                growth = growth[~growth.index.isin(RESIDUAL)]
            growth = growth[(growth > 0).all(axis=1)].copy()
            growth['lis_growth'] = (growth['2022_lis'] - growth['2017_lis']) / growth['2017_lis']
            growth['pwc_growth'] = (growth['2022_pwc'] - growth['2017_pwc']) / growth['2017_pwc']
            # Two-sided tests. Spearman uses average ranks and the t approximation
            # (equivalent to R cor.test(..., exact=FALSE)).
            pearson = stats.pearsonr(growth.pwc_growth, growth.lis_growth)
            spearman = stats.spearmanr(growth.pwc_growth, growth.lis_growth)
            correlations.append({'model': f'M{model}', 'specification': MODELS[model], 'level': level, 'n': len(growth),
                                 'pearson_r': pearson.statistic, 'pearson_p': pearson.pvalue,
                                 'spearman_rho': spearman.statistic, 'spearman_p': spearman.pvalue})
            growth_details.append(growth.reset_index().assign(model=f'M{model}', level=level))
    correlations = pd.DataFrame(correlations)
    p = np.r_[correlations.pearson_p, correlations.spearman_p]
    order = np.argsort(p, kind='stable')
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    correlations['pearson_p_holm'] = adjusted[:len(correlations)]
    correlations['spearman_p_holm'] = adjusted[len(correlations):]
    summary['growth_tests'] = {'tests': len(p), 'holm_significant_at_005': int((adjusted < .05).sum())}
    journal_rows = []
    for row in j.itertuples():
        ref, matched = w[w.journal.eq(row.journal)], links[links.journal.eq(row.journal)]
        journal_rows.append({'journal': row.journal, 'group': row.group, 'criterion': row.criterion,
                             'asjc_codes': '; '.join(row.asjc_codes), 'lis_publications': ref.doi.nunique(),
                             'references': len(ref), 'ml_links': len(matched), 'has_ml': 'Yes' if len(matched) else 'No'})

    print('6/6 Computing snapshot coverage and exporting final tables', flush=True)
    snapshot = pd.read_csv(data / 'pwc_snapshot_dates.csv.gz', dtype=str)
    snapshot['ax'] = snapshot.arxiv_id.map(normalize_arxiv)
    snapshot['year'] = pd.to_numeric(snapshot.date.str[:4], errors='coerce')
    snapshot = snapshot[snapshot.ax.ne('')]
    figure1 = snapshot.groupby('year').size().reset_index(name='records')
    snapshot_ids = set(snapshot.ax)
    arxiv = pd.read_csv(data / 'arxiv_categories.csv.gz', dtype=str)
    denominator, numerator = Counter(), Counter()
    for row in arxiv.itertuples(index=False):
        match = re.match(r'^(\d{2})\d{2}\.', row.id)
        if not match or not 2010 <= 2000 + int(match[1]) <= 2022:
            continue
        for category in set(row.categories.split()) & set(ARXIV_CATEGORIES):
            denominator[category] += 1
            numerator[category] += normalize_arxiv(row.id) in snapshot_ids
    b1 = pd.DataFrame([{'category': category, 'arxiv_records': denominator[category],
                        'matched_pwc_records': numerator[category], 'coverage_pct': numerator[category] / denominator[category] * 100}
                       for category in ARXIV_CATEGORIES])

    for name, rows in annotations['display_tables'].items():
        pd.DataFrame(rows[1:], columns=rows[0]).to_csv(tables_dir / f'{name}.csv', index=False, encoding='utf-8-sig')
    # Appendix A is a curated selection, not a random sample: verify every label.
    task_hierarchy, _, _ = catalogues['task']
    for root, examples in annotations['display_tables']['TableA1_task_examples'][1:]:
        valid = set(task_hierarchy.loc[task_hierarchy.iloc[:, 0].eq(root)].iloc[:, 1].dropna())
        for label in examples.rstrip('…').split(';'):
            assert label.strip() in valid, ('Appendix A1', root, label)
    method_hierarchy, _, _ = catalogues['method']
    for root, parent, examples in annotations['display_tables']['TableA2_method_examples'][1:]:
        valid = set(method_hierarchy.loc[method_hierarchy.iloc[:, 0].eq(root) & method_hierarchy.iloc[:, 1].eq(parent)].iloc[:, 2].dropna())
        assert all(label.strip() in valid for label in examples.split(';'))

    tables = {'Table2_citation_age': age_table, 'Table3_annual_citation_age': pd.DataFrame(annual_age),
              'Table4_task_shares': table4, 'TableB1_arxiv_coverage': b1,
              'TableB2_journals': pd.DataFrame(journal_rows), 'TableB3_sensitivity': pd.DataFrame(boundary_summary),
              'TableB4_annual_shares': pd.DataFrame(boundary_annual).pivot(index='year', columns='model', values='ml_share').reset_index(),
              'TableB5_correlations': correlations}
    plot_data = {'figure1': figure1, 'annual_results': annual, 'figure4_coverage': coverage,
                 'figure5_fields': fields, 'citation_age': age_table, 'figure6': figure6,
                 'figure6_statistics': statistics6, 'figure7': figure7}
    for name, frame in tables.items():
        frame.to_csv(tables_dir / f'{name}.csv', index=False, encoding='utf-8-sig')
    for name, frame in plot_data.items():
        if name != 'citation_age':
            frame.to_csv(figures_data / f'{name}.csv', index=False, encoding='utf-8-sig')
    pd.concat(growth_details, ignore_index=True).to_csv(output / 'growth_by_task.csv', index=False, encoding='utf-8-sig')
    # Publication counts, reference counts and task weights must reconcile.
    assert annual.references.sum() == len(w) and annual.ml_links.sum() == len(links)
    assert sum(row['references'] for row in journal_rows) == len(w)
    assert sum(row['ml_links'] for row in journal_rows) == len(links)
    summary['checks'] = {'unique_citation_links': True, 'task_weight_conserved': True,
                          'annual_totals_reconcile': True, 'journal_totals_reconcile': True,
                          'appendix_examples_in_catalogue': True}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
    (output / 'AppendixC_retrieval_query.txt').write_text(annotations['retrieval_vocabulary']+'\n', encoding='utf8')
    return plot_data, tables, summary
