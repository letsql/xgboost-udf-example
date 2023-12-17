use datafusion::assert_batches_eq;
use datafusion::error::Result;
use datafusion::{execution::options::CsvReadOptions, prelude::SessionContext};
use xgboost_udf_example::register_udfs;

#[tokio::test]
async fn it_onehots() -> Result<()> {
    let ctx = SessionContext::new();
    register_udfs(&ctx);
    ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new())
        .await?;
    let sql = "SELECT onehot(arrow_cast(class, 'Dictionary(Int32, Utf8)')) as class from mushrooms";
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
