extern crate xgboost;

use datafusion::arrow::array::DictionaryArray;
use datafusion::arrow::array::{Array, ArrayRef, StringArray};
use datafusion::arrow::array::{BooleanBuilder, ListBuilder, StringBuilder, StructBuilder};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Int32Type;
use datafusion::arrow::datatypes::{Field, Fields};
use datafusion::error::Result;
use datafusion::logical_expr::{create_udf, Volatility};
use datafusion::physical_plan::functions::make_scalar_function;
use datafusion::prelude::SessionContext;
use std::sync::Arc;
use xgboost::{parameters, Booster, DMatrix};

#[tokio::main]
async fn main() -> Result<()> {
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    Utf8(Option<String>),
}

macro_rules! typed_cast {
    ($array:expr, $index:expr, $ARRAYTYPE:ident, $SCALAR:ident) => {{
        let array = $array.as_any().downcast_ref::<$ARRAYTYPE>().unwrap();
        ScalarValue::$SCALAR(match array.is_null($index) {
            true => None,
            false => Some(array.value($index).into()),
        })
    }};
}

impl ScalarValue {
    pub fn try_from_array(array: &ArrayRef, index: usize) -> Result<Self, String> {
        match array.data_type() {
            DataType::Utf8 => Ok(typed_cast!(array, index, StringArray, Utf8)),
            _ => Err(format!(
                "Unsupported data type {:?} for ScalarValue",
                array.data_type()
            )),
        }
    }
}

fn onehot(args: &[ArrayRef]) -> Result<ArrayRef> {
    let data = args[0]
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap()
        .clone();
    let (key, values) = data.into_parts();
    let values = values.as_any().downcast_ref::<StringArray>().unwrap();

    let new_struct = StructBuilder::from_fields(
        Fields::from(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Boolean, false),
        ]),
        2,
    );
    let mut list_builder = ListBuilder::new(new_struct);
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
        Arc::new(list_type), //vec![DataType::Dictionary(Box::new(DataType::Utf8), Box::new(DataType::Boolean))],
        Volatility::Immutable,
        onehot,
    );
    ctx.register_udf(onehot_udf);
}

#[cfg(test)]
mod test {
    use super::*;
    use datafusion::arrow::util::pretty::print_batches;
    use datafusion::prelude::CsvReadOptions;

    #[tokio::test]
    pub async fn test_recordbatch_downcast() -> Result<()> {
        let ctx = SessionContext::new();
        let _ = ctx
            .register_csv("data", "./data/mushrooms.csv", CsvReadOptions::default())
            .await?;

        let batches = ctx.sql("SELECT class FROM data").await?.collect().await?;
        let array = batches[0].column(0);

        match array.data_type() {
            DataType::Utf8 => {
                let array = array.as_any().downcast_ref::<StringArray>().unwrap();
                println!("{:?}", array);
            }
            _ => println!(
                "Unsupported data type {:?} for ScalarValue",
                array.data_type()
            ),
        }
        Ok(())
    }

    #[tokio::test]
    pub async fn test_scalar_value() -> Result<()> {
        let ctx = SessionContext::new();
        let _ = ctx
            .register_csv("data", "./data/mushrooms.csv", CsvReadOptions::default())
            .await?;
        let batches = ctx.sql("SELECT class FROM data").await?.collect().await?;
        let array = batches[0].column(0);
        let result = ScalarValue::try_from_array(&array, 0).unwrap();

        println!("{:?}", result);
        Ok(())
    }

    #[tokio::test]
    pub async fn test_onehot() -> Result<()> {
        let ctx = SessionContext::new();
        register_udfs(&ctx);
        let _ = ctx
            .register_csv("data", "./data/mushrooms.csv", CsvReadOptions::default())
            .await?;
        let batches = ctx
            .sql("SELECT onehot(arrow_cast(class, 'Dictionary(Int32, Utf8)')) as class FROM data")
            .await?
            .collect()
            .await?;
        print_batches(&batches).unwrap();
        let array = batches[0].column(0);
        assert_eq!(array.len(), 8124);
        Ok(())
    }
}
