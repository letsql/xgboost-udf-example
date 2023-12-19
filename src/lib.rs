use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, BooleanBuilder, DictionaryArray, ListArray, ListBuilder,
    StringArray, StringBuilder, StructArray, StructBuilder,
};
use datafusion::arrow::datatypes::{DataType, Field, Fields, Int32Type};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{create_udf, Volatility};
use datafusion::physical_plan::functions::make_scalar_function;
use datafusion::prelude::SessionContext;
use std::sync::Arc;
use xgboost::DMatrix;

fn onehot(args: &[ArrayRef]) -> Result<ArrayRef> {
    let data = args[0]
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap()
        .clone();
    let (key, values) = data.into_parts();
    let values = values.as_any().downcast_ref::<StringArray>().unwrap();

    let struct_builder = StructBuilder::from_fields(
        Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]),
        2,
    );
    let mut list_builder = ListBuilder::new(struct_builder);
    for i in 0..key.len() {
        for j in 0..values.len() {
            let key_value = key.value(i) as usize;
            let struct_key = values.value(j);
            let struct_value = j == key_value;

            list_builder
                .values()
                .field_builder::<StringBuilder>(0)
                .unwrap()
                .append_value(struct_key);
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
}

pub fn convert_to_native(batch: &RecordBatch) -> Result<(Vec<bool>, usize), DataFusionError> {
    let num_rows = batch.num_rows();
    let array = batch.columns()[0]
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| DataFusionError::Internal("Expected ListArray".to_string()))?;

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

            result.push(boolean_array.value(0));
        } else {
            return Err(DataFusionError::Internal(
                "Null struct in ListArray".to_string(),
            ));
        }
    }
    Ok((result, num_rows))
}

pub fn create_dmatrix(data: Vec<bool>, num_rows: usize) -> Result<DMatrix, DataFusionError> {
    //convert data from bool to Vec<f32>
    let data_transform = data
        .into_iter()
        .map(|x| x as u8 as f32)
        .collect::<Vec<f32>>();
    let dmat = DMatrix::from_dense(&data_transform, num_rows)
        .map_err(|_| DataFusionError::Internal("Failed to create dmatrix".to_string()))?;
    Ok(dmat)
}

#[cfg(test)]
mod test {
    use super::*;
    use datafusion::arrow::array::UInt8Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::assert_batches_eq;
    use datafusion::datasource::MemTable;

    fn create_record_batch() -> Result<RecordBatch> {
        let id_array = UInt8Array::from(vec![1, 2]);
        let account_array = StringArray::from(vec![Some("a"), Some("b")]);

        Ok(RecordBatch::try_new(
            get_schema(),
            vec![Arc::new(id_array), Arc::new(account_array)],
        )
        .unwrap())
    }

    pub fn get_schema() -> SchemaRef {
        SchemaRef::new(Schema::new(vec![
            Field::new("id", DataType::UInt8, false),
            Field::new("class", DataType::Utf8, true),
        ]))
    }

    #[tokio::test]
    pub async fn test_onehot() -> Result<()> {
        let mem_table = MemTable::try_new(get_schema(), vec![vec![create_record_batch()?]])?;
        let ctx = SessionContext::new();
        register_udfs(&ctx);
        ctx.register_table("training", Arc::new(mem_table))?;
        let batches = ctx
            .sql("SELECT onehot(arrow_cast(class, 'Dictionary(Int32, Utf8)')) as class FROM training")
            .await?
            .collect()
            .await?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].columns()[0].len(), 2);
        let expected = [
            r#"+-------------------------------------------------+"#,
            r#"| class                                           |"#,
            r#"+-------------------------------------------------+"#,
            r#"| [{key: a, value: true}, {key: b, value: false}] |"#,
            r#"| [{key: a, value: false}, {key: b, value: true}] |"#,
            r#"+-------------------------------------------------+"#,
        ];
        assert_batches_eq!(expected, &batches);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_onehot_to_vec() -> Result<()> {
        let mem_table = MemTable::try_new(get_schema(), vec![vec![create_record_batch()?]])?;
        let ctx = SessionContext::new();
        register_udfs(&ctx);
        ctx.register_table("training", Arc::new(mem_table))?;
        let batches = ctx
            .sql("SELECT onehot(arrow_cast(class, 'Dictionary(Int32, Utf8)')) as class FROM training")
            .await?
            .collect()
            .await?;

        let (result, num_rows) = convert_to_native(&batches[0])?;
        assert_eq!(result, vec![true, false]);
        assert_eq!(num_rows, 2);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_build_dmatrix() -> Result<()> {
        let mem_table = MemTable::try_new(get_schema(), vec![vec![create_record_batch()?]])?;
        let ctx = SessionContext::new();
        register_udfs(&ctx);
        ctx.register_table("training", Arc::new(mem_table))?;
        let batches = ctx
            .sql("SELECT onehot(arrow_cast(class, 'Dictionary(Int32, Utf8)')) as class FROM training")
            .await?
            .collect()
            .await?;

        let (data, num_rows) = convert_to_native(&batches[0])?;
        let dm = create_dmatrix(data, num_rows)?;
        assert_eq!(dm.shape(), (2, 1));
        Ok(())
    }
}
