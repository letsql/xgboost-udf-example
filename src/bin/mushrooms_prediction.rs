use datafusion::arrow::util::pretty::print_batches;
use datafusion::error::{DataFusionError, Result};
use datafusion::{execution::options::CsvReadOptions, prelude::SessionContext};
use xgboost_udf_example::register_udfs;

#[tokio::main]
async fn main() -> Result<(), DataFusionError> {
    let ctx = SessionContext::new();
    register_udfs(&ctx);
    ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new())
        .await?;
    let sql = "SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape FROM mushrooms";
    let batches = ctx.sql(sql).await?.collect().await?;
    print_batches(&batches)?;

    Ok(())
}

#[tokio::test]
async fn predict() -> Result<(), DataFusionError> {
    let ctx = SessionContext::new();
    register_udfs(&ctx);
    ctx.register_csv("mushrooms", "./data/mushrooms.csv", CsvReadOptions::new())
        .await?;
    let sql = "SELECT predict(cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_root,\
        stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type, population, habitat ) as predictions FROM (SELECT \
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
            FROM mushrooms) data LIMIT 5";
    let batches = ctx.sql(sql).await?.collect().await?;
    print_batches(&batches)?;
    Ok(())
}
