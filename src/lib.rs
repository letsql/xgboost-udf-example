use datafusion::arrow::array::{
    as_dictionary_array, Array, ArrayRef, BooleanArray, BooleanBuilder, DictionaryArray,
    Float32Array, ListArray, ListBuilder, StringArray, StringBuilder, StructArray, StructBuilder,
};
use datafusion::arrow::datatypes::{DataType, Field, Fields, Int32Type};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{create_udf, Volatility};
use datafusion::physical_plan::functions::make_scalar_function;
use datafusion::prelude::SessionContext;
use std::sync::Arc;
use xgboost::{Booster, DMatrix};

fn onehot(args: &[ArrayRef]) -> Result<ArrayRef> {
    let data: &DictionaryArray<Int32Type> = as_dictionary_array::<_>(&args[0]);
    let key = data.keys();
    let values = data.values();

    let values = values.as_any().downcast_ref::<StringArray>().unwrap();

    let struct_builder = StructBuilder::from_fields(
        Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]),
        2,
    );

    let mut list_builder = ListBuilder::new(struct_builder);
    for key_value in key.iter() {
        for (j, struct_key) in values.iter().enumerate() {
            let struct_value = j == key_value.unwrap() as usize;
            list_builder
                .values()
                .field_builder::<StringBuilder>(0)
                .unwrap()
                .append_value(struct_key.unwrap());
            list_builder
                .values()
                .field_builder::<BooleanBuilder>(1)
                .unwrap()
                .append_value(struct_value);
            list_builder.values().append(true);
        }
        list_builder.append(true);
    }
    let list_array = list_builder.finish();
    Ok(Arc::new(list_array))
}

pub fn register_udfs(ctx: &SessionContext) {
    let onehot = make_scalar_function(onehot);
    let struct_type = DataType::Struct(Fields::from(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Boolean, false),
    ]));

    let list_field = Field::new("item", struct_type, true);
    let list_type = DataType::List(Arc::new(list_field));
    let onehot_udf = create_udf(
        "onehot",
        vec![DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        )],
        Arc::new(list_type),
        Volatility::Immutable,
        onehot,
    );

    ctx.register_udf(onehot_udf);
    let predict = make_scalar_function(predict);
    let struct_type = DataType::Struct(Fields::from(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new("value", DataType::Boolean, false),
    ]));

    let list_field = Field::new("item", struct_type, true);
    let list_type = DataType::List(Arc::new(list_field));
    let predict_udf = create_udf(
        "predict",
        vec![list_type.clone(); 21],
        Arc::new(DataType::Float32),
        Volatility::Immutable,
        predict,
    );

    ctx.register_udf(predict_udf);
}

pub fn convert_to_native(
    batch: &ArrayRef,
    column: usize,
) -> Result<(Vec<bool>, usize), DataFusionError> {
    let array = batch
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| DataFusionError::Internal("Expected ListArray".to_string()))?;

    let num_rows = array.len();
    let mut result = Vec::new();

    for maybe_struct in array.iter() {
        if let Some(struct_val) = maybe_struct {
            let struct_array = struct_val
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| DataFusionError::Internal("Expected StructArray".to_string()))?;

            let boolean_array = struct_array
                .column(1)
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| DataFusionError::Internal("Expected BooleanArray".to_string()))?;

            result.push(boolean_array.value(column));
        } else {
            return Err(DataFusionError::Internal(
                "Null struct in ListArray".to_string(),
            ));
        }
    }
    Ok((result, num_rows))
}

fn to_dense(batch: &ArrayRef) -> Result<(Vec<bool>, usize, Vec<String>), DataFusionError> {
    let array = batch
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| DataFusionError::Internal("Expected ListArray".to_string()))?;

    let mut result = Vec::new();
    let first_item = array.value(0);
    let struct_array = first_item
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| DataFusionError::Internal("Expected StructArray".to_string()))?;
    let string_array = struct_array
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| DataFusionError::Internal("Expected StringArray".to_string()))?;

    let mut num = 0;
    for i in 0..string_array.len() {
        let (single, num_rows) = convert_to_native(batch, i)?;
        result.push(single);
        num = num_rows;
    }
    let flattened = result.into_iter().flatten().collect();

    let dim_names = string_array
        .into_iter()
        .map(|x| x.unwrap().to_string())
        .collect();
    Ok((flattened, num, dim_names))
}

fn records_to_dense(
    batch: &RecordBatch,
) -> Result<(Vec<bool>, usize, Vec<String>), DataFusionError> {
    let mut result = Vec::new();
    let mut num_rows = 0;
    let mut dim_names = Vec::new();

    for i in 0..batch.num_columns() {
        let (result_col, num_rows_col, dim_names_col) = to_dense(batch.column(i))?;
        result.extend(result_col);
        num_rows = num_rows_col;
        dim_names.extend(dim_names_col);
    }
    Ok((result, num_rows, dim_names))
}

pub fn create_dmatrix(data: &RecordBatch) -> Result<DMatrix, DataFusionError> {
    let (data, num_rows, _) = records_to_dense(data)?;
    let data_transform = data
        .into_iter()
        .map(|x| x as u8 as f32)
        .collect::<Vec<f32>>();
    let dmat = DMatrix::from_dense(&data_transform, num_rows)
        .map_err(|_| DataFusionError::Internal("Failed to create dmatrix".to_string()))?;
    Ok(dmat)
}

fn predict(args: &[ArrayRef]) -> Result<ArrayRef> {
    let mut result = Vec::new();
    let mut num_rows = 0;
    let mut dim_names = Vec::new();
    println!("args len: {}", args.len());

    for arg in args {
        let (result_col, num_rows_col, dim_names_col) = to_dense(arg)?;
        result.extend(result_col);
        num_rows = num_rows_col;
        dim_names.extend(dim_names_col);
    }
    let data_transform = result
        .into_iter()
        .map(|x| x as u8 as f32)
        .collect::<Vec<f32>>();
    let dmat = DMatrix::from_dense(&data_transform, num_rows).unwrap();
    println!("dmat shape: {:?}", dmat.shape());
    let booster = Booster::load("model.xgb").unwrap();
    let result = Float32Array::from(booster.predict(&dmat).unwrap());

    Ok(Arc::new(result))
}

#[cfg(test)]
mod test {
    use super::*;
    use datafusion::arrow::array::{ArrayRef, StringDictionaryBuilder};
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::arrow::record_batch::RecordBatch;

    #[tokio::test]
    pub async fn test_onehot() -> Result<()> {
        let mut builder = StringDictionaryBuilder::<Int32Type>::new();
        builder.append("a").unwrap();
        builder.append("b").unwrap();
        let dict = Arc::new(builder.finish()).clone();
        let result = onehot(&[dict])?;
        assert_eq!(result.len(), 2);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_onehot_to_vec() -> Result<()> {
        let fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]);
        let struct_builder = StructBuilder::from_fields(fields, 2);
        let mut list_builder = ListBuilder::new(struct_builder);
        list_builder
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value("a");
        list_builder
            .values()
            .field_builder::<BooleanBuilder>(1)
            .unwrap()
            .append_value(true);
        list_builder.values().append(true);
        list_builder.append(true);
        let array = Arc::new(list_builder.finish()).clone() as ArrayRef;

        let (result, num_rows) = convert_to_native(&array, 0)?;
        assert_eq!(result, vec![true]);
        assert_eq!(num_rows, 1);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_onehot_to_dense() -> Result<()> {
        let fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]);
        let struct_builder = StructBuilder::from_fields(fields, 2);
        let mut list_builder = ListBuilder::new(struct_builder);
        list_builder
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value("a");
        list_builder
            .values()
            .field_builder::<BooleanBuilder>(1)
            .unwrap()
            .append_value(true);
        list_builder.values().append(true);
        list_builder
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value("b");
        list_builder
            .values()
            .field_builder::<BooleanBuilder>(1)
            .unwrap()
            .append_value(false);
        list_builder.values().append(true);
        list_builder.append(true);
        let array = Arc::new(list_builder.finish()).clone() as ArrayRef;

        let (result, num_rows, dim_names) = to_dense(&array)?;
        assert_eq!(result.len(), 2);
        assert_eq!(num_rows, 1);
        assert_eq!(dim_names, vec!["a", "b"]);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_records_to_dense() -> Result<()> {
        let struct_type = DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]));

        let list_field = Field::new("item", struct_type, true);
        let list_type = DataType::List(Arc::new(list_field));
        let schema = SchemaRef::new(Schema::new(vec![Field::new(
            "class",
            list_type.clone(),
            true,
        )]));

        let fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]);
        let struct_builder = StructBuilder::from_fields(fields, 2);
        let mut list_builder = ListBuilder::new(struct_builder);
        list_builder
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value("a");
        list_builder
            .values()
            .field_builder::<BooleanBuilder>(1)
            .unwrap()
            .append_value(true);
        list_builder.values().append(true);
        list_builder.append(true);

        let record_batch = RecordBatch::try_new(schema, vec![Arc::new(list_builder.finish())])?;

        let (result, num_rows, dim_names) = records_to_dense(&record_batch)?;
        assert_eq!(result.len(), 1);
        assert_eq!(num_rows, 1);
        assert_eq!(dim_names, vec!["a"]);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_build_dmatrix() -> Result<()> {
        let struct_type = DataType::Struct(Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]));

        let list_field = Field::new("item", struct_type, true);
        let list_type = DataType::List(Arc::new(list_field));
        let schema = SchemaRef::new(Schema::new(vec![Field::new(
            "class",
            list_type.clone(),
            true,
        )]));

        let fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]);
        let struct_builder = StructBuilder::from_fields(fields, 2);
        let mut list_builder = ListBuilder::new(struct_builder);
        list_builder
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value("a");
        list_builder
            .values()
            .field_builder::<BooleanBuilder>(1)
            .unwrap()
            .append_value(true);
        list_builder.values().append(true);
        list_builder.append(true);

        let record_batch = RecordBatch::try_new(schema, vec![Arc::new(list_builder.finish())])?;

        let dm = create_dmatrix(&record_batch)?;
        assert_eq!(dm.shape(), (1, 1));
        Ok(())
    }

    #[tokio::test]
    pub async fn test_predict() -> Result<()> {
        let fields = Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]);
        let struct_builder = StructBuilder::from_fields(fields, 2);
        let mut list_builder = ListBuilder::new(struct_builder);
        list_builder
            .values()
            .field_builder::<StringBuilder>(0)
            .unwrap()
            .append_value("a");
        list_builder
            .values()
            .field_builder::<BooleanBuilder>(1)
            .unwrap()
            .append_value(true);
        list_builder.values().append(true);
        list_builder.append(true);
        let array = Arc::new(list_builder.finish());

        let f0 = array.clone();
        let f1 = array.clone();
        let f2 = array.clone();
        let f3 = array.clone();

        let _result = predict(&[f0, f1, f2, f3])?;

        Ok(())
    }
}
