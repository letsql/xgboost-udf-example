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
    let sql = "SELECT predict(cap_shape,cap_surface,cap_color,bruises) as predictions \
               FROM (SELECT onehot(arrow_cast(cap_shape, 'Dictionary(Int32, Utf8)')) as cap_shape, \
                            onehot(arrow_cast(cap_surface, 'Dictionary(Int32, Utf8)')) as cap_surface, \
                            onehot(arrow_cast(cap_color, 'Dictionary(Int32, Utf8)')) as cap_color, \
                            onehot(arrow_cast(bruises, 'Dictionary(Int32, Utf8)')) as bruises FROM mushrooms) \
               data";
    let batches = ctx.sql(sql).await?.collect().await?;
    print_batches(&batches)?;

    Ok(())
}
