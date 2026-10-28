# Day 1 data audit

| file | rows | cols | size KB | status |
|---|---|---|---|---|
| data/raw/aaer/aaer_rows_raw.csv | 3377 | 4 | 555.2 | ok |
| data/raw/macro/worldbank_macro_raw.csv | 11 | 5 | 0.7 | ok |
| data/raw/sec_edgar/company_list_all.csv | 10430 | 3 | 388.3 | ok |
| data/raw/sec_edgar/company_list_sample.csv | 250 | 3 | 8.0 | ok |
| data/raw/sec_edgar/keep_companies.csv | 0 | 3 | 0.0 | ok |

## data/raw/aaer/aaer_rows_raw.csv


| column | dtype | % missing | distinct |
|---|---|---|---|
| page_no | int64 | 0.0 | 34 |
| table_no | int64 | 0.0 | 1 |
| cells | str | 0.0 | 3344 |
| link | str | 0.0 | 3377 |

## data/raw/macro/worldbank_macro_raw.csv


| column | dtype | % missing | distinct |
|---|---|---|---|
| country | str | 0.0 | 1 |
| year | int64 | 0.0 | 11 |
| gdp_growth_pct | float64 | 0.0 | 11 |
| credit_to_gdp_pct | float64 | 0.0 | 11 |
| npl_ratio_pct | float64 | 0.0 | 11 |

## data/raw/sec_edgar/company_list_all.csv


| column | dtype | % missing | distinct |
|---|---|---|---|
| cik | int64 | 0.0 | 8023 |
| ticker | str | 0.0 | 10429 |
| company_name | str | 0.0 | 8022 |

## data/raw/sec_edgar/company_list_sample.csv


| column | dtype | % missing | distinct |
|---|---|---|---|
| cik | int64 | 0.0 | 250 |
| ticker | str | 0.0 | 250 |
| company_name | str | 0.0 | 250 |

## data/raw/sec_edgar/keep_companies.csv


| column | dtype | % missing | distinct |
|---|---|---|---|
| cik | object | nan | 0 |
| ticker | object | nan | 0 |
| company_name | object | nan | 0 |