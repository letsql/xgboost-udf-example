use datafusion::error::Result;
use datafusion::{execution::options::CsvReadOptions, prelude::SessionContext};
use xgboost_udf_example::register_udfs;

use criterion::{criterion_group, criterion_main, Criterion};

use tokio::runtime::Runtime;

async fn run_sql(ctx: &SessionContext, sql: &str) -> Result<()> {
    let _batches = ctx.sql(sql).await?.collect().await?;
    Ok(())
}

fn bench_mushrooms_prediction(c: &mut Criterion) {
    let ctx = SessionContext::new();
    let _ = ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new());
    register_udfs(&ctx);
    let sql = "SELECT predict(cap_shape,cap_surface,cap_color,bruises) as predictions FROM (SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape, onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises FROM mushrooms) data";
    

    c.bench_function("mushrooms_prediction", |b| {
        b.to_async(Runtime::new().unwrap()).iter( || 
            run_sql(&ctx, &sql))
    });
}

fn bench_mushrooms_onehot(c: &mut Criterion) {
    let ctx = SessionContext::new();
    let _ = ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new());
    register_udfs(&ctx);
    let sql = "SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape, onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises FROM mushrooms";
    

    c.bench_function("mushrooms_prediction", |b| {
        b.to_async(Runtime::new().unwrap()).iter( || 
            run_sql(&ctx, &sql))
    });
}

criterion_group!(benches, bench_mushrooms_prediction, bench_mushrooms_onehot);
criterion_main!(benches);
