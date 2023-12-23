use datafusion::error::Result;
use datafusion::{execution::options::CsvReadOptions, prelude::SessionContext};
use xgboost_udf_example::register_udfs;

use criterion::{criterion_group, criterion_main, Criterion};

use tokio::runtime::Runtime;

async fn score() -> Result<()> {
    let ctx = SessionContext::new();
    register_udfs(&ctx);
    ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new())
        .await?;
    let sql = "SELECT predict(cap_shape,cap_surface,cap_color,bruises) as predictions FROM (SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape, onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises FROM mushrooms) data";
    let _batches = ctx.sql(sql).await?.collect().await?;
    Ok(())
}

fn bench_mushrooms_prediction(c: &mut Criterion) {
    c.bench_function("mushrooms_prediction", move |b| {
        b.to_async(Runtime::new().unwrap()).iter(|| score())
    });
}

criterion_group!(benches, bench_mushrooms_prediction);
criterion_main!(benches);
