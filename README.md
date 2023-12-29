# [WIP] DATAFUSION XGBOOST PREDICTION EXAMPLE
This is an example of using DataFusion to add a UDF to do one-hot encoding and run an XGBoost predict UDF. 

## SQL

```sql
SELECT predict(cap_shape,cap_surface,cap_color,bruises) as predictions FROM 
  (SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape, 
          onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, 
          onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, 
          onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises 
  FROM mushrooms);
```
The predict UDF loads a already trained XGBoost model from disk. 

## BENCHMARKS
This benchmark converts 4 columns into 22 and scores 8124 rows from Mushrooms datasets and outputs `RecordBatch`.

|      | onehot UDF |
|------|------------|
| Time | 48.59 µs   |

As a naive comparison, to do `get_dummies` in Python/pandas land is around 40x slower with ~1.97ms


