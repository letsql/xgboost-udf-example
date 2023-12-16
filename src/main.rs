extern crate xgboost;

use datafusion::arrow::array::BooleanArray;
use datafusion::arrow::array::DictionaryArray;
use datafusion::arrow::array::StructArray;
use datafusion::arrow::array::{Array, ArrayRef, StringArray};
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Int32Type;
use datafusion::arrow::datatypes::{Field, Fields};
use datafusion::arrow::util::pretty::print_batches;
use datafusion::common::DataFusionError;
use datafusion::error::Result;
use datafusion::logical_expr::{create_udf, Volatility};
use datafusion::physical_plan::functions::make_scalar_function;
use datafusion::prelude::{CsvReadOptions, SessionContext};
use std::sync::Arc;
use xgboost::{parameters, Booster, DMatrix};

#[tokio::main]
async fn main() -> Result<()> {
    Ok(())
}

fn xgboost_example() {
    // initialise logging, run with e.g. RUST_LOG=xgboost=debug to see more details

    // load train and test matrices from text files (in LibSVM format).
    println!("Loading train and test matrices...");
    let dtrain = DMatrix::load("agaricus.txt.train").unwrap();
    println!("Train matrix: {}x{}", dtrain.num_rows(), dtrain.num_cols());
    let dtest = DMatrix::load("agaricus.txt.test").unwrap();
    println!("Test matrix: {}x{}", dtest.num_rows(), dtest.num_cols());

    let dmatrix_dense = DMatrix::from_dense(&[1.0, 2.0, 3.0, 4.0], 2).unwrap();
    println!(
        "Test matrix: {}x{}",
        dmatrix_dense.num_rows(),
        dmatrix_dense.num_cols()
    );

    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .build()
        .unwrap();

    // configure the tree-based learning model's parameters
    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
        .max_depth(2)
        .eta(1.0)
        .build()
        .unwrap();

    // overall configuration for Booster
    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(true)
        .build()
        .unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = [(&dtest, "test"), (&dtrain, "train")];

    // overall configuration for training/evaluation
    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain) // dataset to train with
        .boost_rounds(2) // number of training iterations
        .booster_params(booster_params) // model parameters
        .evaluation_sets(Some(&evaluation_sets)) // optional datasets to evaluate against in each iteration
        .build()
        .unwrap();

    // train booster model, and print evaluation metrics
    println!("\nTraining tree booster...");
    let booster = Booster::train(&training_params).unwrap();

    // get predictions probabilities for given matrix
    let preds = booster.predict(&dtest).unwrap();

    // get predicted labels for each test example (i.e. 0 or 1)
    println!("\nChecking predictions...");
    let labels = dtest.get_labels().unwrap();
    println!(
        "First 3 predicted labels: {} {} {}",
        labels[0], labels[1], labels[2]
    );

    // print error rate
    let num_correct: usize = preds.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).sum();
    println!(
        "error={} ({}/{} correct)",
        num_correct as f32 / preds.len() as f32,
        num_correct,
        preds.len()
    );

    // save and load model file
    println!("\nSaving and loading Booster model...");
    booster.save("xgb.model").unwrap();
    let booster = Booster::load("xgb.model").unwrap();
    let preds2 = booster.predict(&dtest).unwrap();
    assert_eq!(preds, preds2);

    // save and load data matrix file
    println!("\nSaving and loading matrix data...");
    dtest.save("test.dmat").unwrap();
    let dtest2 = DMatrix::load("test.dmat").unwrap();
    assert_eq!(booster.predict(&dtest2).unwrap(), preds);

    // error handling example
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
    //let data = args[0]
    //    .as_any()
    //    .downcast_ref::<DictionaryArray<Int32Type>>()
    //    .unwrap();
    //
    //let (key, values) = data.clone().into_parts();
    //let values = values.as_any().downcast_ref::<StringArray>().unwrap();
    //let mut complete_result: Vec<Vec<bool>> = vec![];
    //let mut fields: Vec<Field> = vec![];

    //for i in 0..values.len() {
    //    let value = values.value(i);
    //    fields.push(Field::new(value, DataType::Boolean, false));
    //    let mut result: Vec<bool> = vec![];

    //    for k in 0..key.len() {
    //        if value == values.value(key.value(k) as usize) {
    //            result.push(true);
    //        } else {
    //            result.push(false);
    //        }
    //    }
    //    complete_result.push(result);
    //}
    //let mut int32_arrays: Vec<ArrayRef> = vec![];
    //for i in 0..complete_result.len() {
    //    int32_arrays.push(Arc::new(BooleanArray::from(complete_result[i].clone())) as ArrayRef);
    //}

    //let field_array_pairs: Vec<(Arc<Field>, ArrayRef)> = fields
    //    .into_iter()
    //    .map(Arc::new)
    //    .zip(int32_arrays.into_iter())
    //    .collect();

    //let struct_array = StructArray::from(field_array_pairs);
    let data = args[0]
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .unwrap()
        .clone();

    let (key, values) = data.into_parts();
    let values = values.as_any().downcast_ref::<StringArray>().unwrap();

    let field_array_pairs: Vec<(Arc<Field>, ArrayRef)> = values
        .iter()
        .map(|value_option| {
            let value = value_option.unwrap(); // Assuming value is always Some
            let field = Arc::new(Field::new(value, DataType::Boolean, false));

            let boolean_values: Vec<bool> = key
                .iter()
                .map(|key_option| {
                    let key_value = key_option.unwrap(); // Assuming key is always Some
                    value == values.value(key_value as usize)
                })
                .collect();

            let array = Arc::new(BooleanArray::from(boolean_values)) as ArrayRef;
            (field, array)
        })
        .collect();

    let struct_array = StructArray::from(field_array_pairs);

    Ok(Arc::new(struct_array))
}

pub fn register_udfs(ctx: &SessionContext) {
    let onehot = make_scalar_function(onehot);
    let onehot_udf = create_udf(
        "onehot",
        vec![DataType::Dictionary(
            Box::new(DataType::Int32),
            Box::new(DataType::Utf8),
        )],
        Arc::new(DataType::Struct(Fields::from(vec![
            Field::new("p", DataType::Boolean, false),
            Field::new("e", DataType::Boolean, false),
        ]))),
        Volatility::Immutable,
        onehot,
    );
    ctx.register_udf(onehot_udf);
}

#[cfg(test)]
mod test {
    use super::*;

    #[tokio::test]
    pub async fn test() -> Result<()> {
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
    pub async fn test_pretty() -> Result<()> {
        let ctx = SessionContext::new();
        let _ = ctx
            .register_csv("data", "./data/mushrooms.csv", CsvReadOptions::default())
            .await?;
        let batches = ctx
            .sql("SELECT arrow_cast(class, 'Dictionary(Int32, Utf8)') FROM data")
            .await?
            .collect()
            .await?;

        print_batches(&batches).unwrap();
        Ok(())
    }

    #[tokio::test]
    pub async fn test_onehot_recordbatch() -> Result<()> {
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
