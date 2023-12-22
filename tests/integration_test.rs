use datafusion::assert_batches_eq;
use datafusion::error::Result;
use datafusion::{execution::options::CsvReadOptions, prelude::SessionContext};
use xgboost::{parameters, Booster};
use xgboost_udf_example::{convert_to_native, create_dmatrix, register_udfs};

#[tokio::test]
async fn it_onehots() -> Result<()> {
    let ctx = SessionContext::new();
    register_udfs(&ctx);
    ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new())
        .await?;
    let sql = "SELECT onehot(arrow_cast(class, 'Dictionary(Int32, Utf8)')) as class FROM mushrooms";
    let batches = ctx.sql(sql).await?.collect().await?;

    let expected = vec![
        r#"+-------------------------------------------------+"#,
        r#"| class                                           |"#,
        r#"+-------------------------------------------------+"#,
        r#"| [{key: p, value: true}, {key: e, value: false}] |"#,
        r#"| [{key: p, value: false}, {key: e, value: true}] |"#,
        r#"| [{key: p, value: false}, {key: e, value: true}] |"#,
        r#"| [{key: p, value: true}, {key: e, value: false}] |"#,
        r#"| [{key: p, value: false}, {key: e, value: true}] |"#,
        r#"+-------------------------------------------------+"#,
    ];

    assert_eq!(batches[0].column(0).len(), 8124);
    assert_batches_eq!(expected, &[batches[0].slice(0, 5)]);

    Ok(())
}

#[tokio::test]
async fn it_creates_dmatrix() -> Result<()> {
    let ctx = SessionContext::new();
    register_udfs(&ctx);
    ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new())
        .await?;
    // how do I make this string multiline?
    let sql = "SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shap, onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises FROM mushrooms";
    let batches = ctx.sql(sql).await?.collect().await?;
    let dmat = create_dmatrix(&batches[0])?;
    assert_eq!(dmat.shape(), (8124, 22));

    Ok(())
}

#[tokio::test]
async fn it_trains_a_model() -> Result<()> {
    let ctx = SessionContext::new();
    register_udfs(&ctx);
    ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new())
        .await?;
    // TODO: use raw string oneday
    let sql = "SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shap, onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises FROM mushrooms";
    let batches = ctx.sql(sql).await?.collect().await?;
    let mut dmat = create_dmatrix(&batches[0])?;
    let sql = "SELECT onehot(arrow_cast(class, 'Dictionary(Int32, Utf8)')) as class FROM mushrooms";
    let classes = ctx.sql(sql).await?.collect().await?;
    let (classes, _) = convert_to_native(&classes[0].column(0), 1)?;
    let data_transform = classes
        .into_iter()
        .map(|x| x as u8 as f32)
        .collect::<Vec<f32>>();

    dmat.set_labels(&data_transform).unwrap();
    let evaluation_sets = &[(&dmat, "train")];

    // specify overall training setup
    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dmat)
        .evaluation_sets(Some(evaluation_sets))
        .build()
        .unwrap();

    // train model, and print evaluation data
    let bst = Booster::train(&training_params).unwrap();

    println!("{:?}", bst.predict(&dmat.slice(&[1, 5]).unwrap()).unwrap());
    bst.save("model.xgb").unwrap();
    Ok(())
}

#[tokio::test]
async fn it_predicts() -> Result<()> {
    let ctx = SessionContext::new();
    register_udfs(&ctx);
    ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new())
        .await?;
    let sql = "SELECT predict(cap_shape,cap_surface,cap_color,bruises) as predictions FROM (SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape, onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises FROM mushrooms LIMIT 5) data";
    let _batches = ctx.sql(sql).await?.collect().await?;
    let _ = datafusion::arrow::util::pretty::print_batches(&_batches);

    assert_eq!(_batches[0].column(0).len(), 5);
    Ok(())
}
