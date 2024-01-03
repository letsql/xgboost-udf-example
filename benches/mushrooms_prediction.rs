use datafusion::error::Result;
use datafusion::{execution::options::CsvReadOptions, prelude::SessionContext};
use xgboost_udf_example::register_udfs;

use criterion::{criterion_group, criterion_main, Criterion};

use tokio::runtime::Runtime;

async fn run_sql(ctx: &SessionContext, sql: &str) -> Result<()> {
    let _batches = ctx.sql(sql).await?.collect().await?;
    Ok(())
}

fn bench_mushrooms_arrow_cast(c: &mut Criterion) {
    let ctx = SessionContext::new();
    let _ = ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new());
    register_udfs(&ctx);
    let sql = "SELECT \
                    (arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape,\
                    (arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, \
                    (arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, \
                    (arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises, \
                    (arrow_cast(odor, 'Dictionary(Int32, Utf8)')) as odor, \
                    (arrow_cast(gill_attachment, 'Dictionary(Int32, Utf8)')) as gill_attachment, \
                    (arrow_cast(gill_spacing, 'Dictionary(Int32, Utf8)')) as gill_spacing, \
                    (arrow_cast(gill_size, 'Dictionary(Int32, Utf8)')) as gill_size, \
                    (arrow_cast(gill_color, 'Dictionary(Int32, Utf8)')) as gill_color, \
                    (arrow_cast(stalk_shape, 'Dictionary(Int32, Utf8)')) as stalk_shape, \
                    (arrow_cast(stalk_root, 'Dictionary(Int32, Utf8)')) as stalk_root, \
                    (arrow_cast(stalk_surface_above_ring, 'Dictionary(Int32, Utf8)')) as stalk_surface_above_ring, \
                    (arrow_cast(stalk_surface_below_ring, 'Dictionary(Int32, Utf8)')) as stalk_surface_below_ring, \
                    (arrow_cast(stalk_color_above_ring, 'Dictionary(Int32, Utf8)')) as stalk_color_above_ring, \
                    (arrow_cast(stalk_color_below_ring, 'Dictionary(Int32, Utf8)')) as stalk_color_below_ring, \
                    (arrow_cast(veil_type, 'Dictionary(Int32, Utf8)')) as veil_type, \
                    (arrow_cast(veil_color, 'Dictionary(Int32, Utf8)')) as veil_color, \
                    (arrow_cast(ring_number, 'Dictionary(Int32, Utf8)')) as ring_number, \
                    (arrow_cast(ring_type, 'Dictionary(Int32, Utf8)')) as ring_type, \
                    (arrow_cast(population, 'Dictionary(Int32, Utf8)')) as population, \
                    (arrow_cast(habitat, 'Dictionary(Int32, Utf8)')) as habitat \
                    FROM mushrooms";

    c.bench_function("mushrooms_arrow_cast", |b| {
        b.to_async(Runtime::new().unwrap())
            .iter(|| run_sql(&ctx, &sql))
    });
}

fn bench_mushrooms_prediction(c: &mut Criterion) {
    let ctx = SessionContext::new();
    let _ = ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new());
    register_udfs(&ctx);
    let sql = "SELECT predict(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_root,\
        stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type, population, habitat ) FROM (SELECT \
                    onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape,\
                    onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, \
                    onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, \
                    onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises, \
                    onehot(arrow_cast(odor, 'Dictionary(Int32, Utf8)')) as odor, \
                    onehot(arrow_cast(gill_attachment, 'Dictionary(Int32, Utf8)')) as gill_attachment, \
                    onehot(arrow_cast(gill_spacing, 'Dictionary(Int32, Utf8)')) as gill_spacing, \
                    onehot(arrow_cast(gill_size, 'Dictionary(Int32, Utf8)')) as gill_size, \
                    onehot(arrow_cast(gill_color, 'Dictionary(Int32, Utf8)')) as gill_color, \
                    onehot(arrow_cast(stalk_shape, 'Dictionary(Int32, Utf8)')) as stalk_shape, \
                    onehot(arrow_cast(stalk_root, 'Dictionary(Int32, Utf8)')) as stalk_root, \
                    onehot(arrow_cast(stalk_surface_above_ring, 'Dictionary(Int32, Utf8)')) as stalk_surface_above_ring, \
                    onehot(arrow_cast(stalk_surface_below_ring, 'Dictionary(Int32, Utf8)')) as stalk_surface_below_ring, \
                    onehot(arrow_cast(stalk_color_above_ring, 'Dictionary(Int32, Utf8)')) as stalk_color_above_ring, \
                    onehot(arrow_cast(stalk_color_below_ring, 'Dictionary(Int32, Utf8)')) as stalk_color_below_ring, \
                    onehot(arrow_cast(veil_type, 'Dictionary(Int32, Utf8)')) as veil_type, \
                    onehot(arrow_cast(veil_color, 'Dictionary(Int32, Utf8)')) as veil_color, \
                    onehot(arrow_cast(ring_number, 'Dictionary(Int32, Utf8)')) as ring_number, \
                    onehot(arrow_cast(ring_type, 'Dictionary(Int32, Utf8)')) as ring_type, \
                    onehot(arrow_cast(population, 'Dictionary(Int32, Utf8)')) as population, \
                    onehot(arrow_cast(habitat, 'Dictionary(Int32, Utf8)')) as habitat \
            FROM mushrooms) data";

    c.bench_function("mushrooms_predict", |b| {
        b.to_async(Runtime::new().unwrap())
            .iter(|| run_sql(&ctx, &sql))
    });
}

fn bench_mushrooms_onehot(c: &mut Criterion) {
    let ctx = SessionContext::new();
    let _ = ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new());
    register_udfs(&ctx);
    let sql = "SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape,\
                    onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, \
                    onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, \
                    onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises, \
                    onehot(arrow_cast(odor, 'Dictionary(Int32, Utf8)')) as odor, \
                    onehot(arrow_cast(gill_attachment, 'Dictionary(Int32, Utf8)')) as gill_attachment, \
                    onehot(arrow_cast(gill_spacing, 'Dictionary(Int32, Utf8)')) as gill_spacing, \
                    onehot(arrow_cast(gill_size, 'Dictionary(Int32, Utf8)')) as gill_size, \
                    onehot(arrow_cast(gill_color, 'Dictionary(Int32, Utf8)')) as gill_color, \
                    onehot(arrow_cast(stalk_shape, 'Dictionary(Int32, Utf8)')) as stalk_shape, \
                    onehot(arrow_cast(stalk_root, 'Dictionary(Int32, Utf8)')) as stalk_root, \
                    onehot(arrow_cast(stalk_surface_above_ring, 'Dictionary(Int32, Utf8)')) as stalk_surface_above_ring, \
                    onehot(arrow_cast(stalk_surface_below_ring, 'Dictionary(Int32, Utf8)')) as stalk_surface_below_ring, \
                    onehot(arrow_cast(stalk_color_above_ring, 'Dictionary(Int32, Utf8)')) as stalk_color_above_ring, \
                    onehot(arrow_cast(stalk_color_below_ring, 'Dictionary(Int32, Utf8)')) as stalk_color_below_ring, \
                    onehot(arrow_cast(veil_type, 'Dictionary(Int32, Utf8)')) as veil_type, \
                    onehot(arrow_cast(veil_color, 'Dictionary(Int32, Utf8)')) as veil_color, \
                    onehot(arrow_cast(ring_number, 'Dictionary(Int32, Utf8)')) as ring_number, \
                    onehot(arrow_cast(ring_type, 'Dictionary(Int32, Utf8)')) as ring_type, \
                    onehot(arrow_cast(population, 'Dictionary(Int32, Utf8)')) as population, \
                    onehot(arrow_cast(habitat, 'Dictionary(Int32, Utf8)')) as habitat \
                    FROM mushrooms";

    c.bench_function("mushrooms_onehot", |b| {
        b.to_async(Runtime::new().unwrap())
            .iter(|| run_sql(&ctx, &sql))
    });
}

fn bench_mushrooms_read(c: &mut Criterion) {
    let ctx = SessionContext::new();
    let _ = ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new());
    register_udfs(&ctx);
    let sql = "SELECT * FROM mushrooms";
    c.bench_function("mushrooms_read", |b| {
        b.to_async(Runtime::new().unwrap())
            .iter(|| run_sql(&ctx, &sql))
    });
}

criterion_group!(
    benches,
    bench_mushrooms_prediction,
    bench_mushrooms_onehot,
    bench_mushrooms_arrow_cast,
    bench_mushrooms_read
);
criterion_main!(benches);
