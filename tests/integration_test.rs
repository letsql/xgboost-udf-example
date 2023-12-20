use datafusion::assert_batches_eq;
use datafusion::error::Result;
use datafusion::{execution::options::CsvReadOptions, prelude::SessionContext};
use xgboost_udf_example::{create_dmatrix, register_udfs};

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
