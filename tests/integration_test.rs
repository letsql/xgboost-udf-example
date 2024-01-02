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
    let sql = "SELECT \
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
              FROM mushrooms";
    let batches = ctx.sql(sql).await?.collect().await?;
    println!("batches len: {}", batches.len());
    let mut dmat = create_dmatrix(&batches[0])?;
    let sql = "SELECT onehot(arrow_cast(class, 'Dictionary(Int32, Utf8)')) as class FROM mushrooms";
    let classes = ctx.sql(sql).await?.collect().await?;
    println!("batches len: {}", batches.len());
    let (classes, _) = convert_to_native(&classes[0].column(0), 1)?;
    let data_transform = classes
        .into_iter()
        .map(|x| x as u8 as f32)
        .collect::<Vec<f32>>();

    dmat.set_labels(&data_transform).unwrap();
    // configure objectives, metrics, etc.
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::BinaryLogistic)
        .build().unwrap();

    // configure the tree-based learning model's parameters
    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
            .max_depth(6)
            .eta(0.1)
            .build().unwrap();

    // overall configuration for Booster
    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(true)
        .build().unwrap();

    // specify datasets to evaluate against during training
    let evaluation_sets = [(&dmat, "train")];

    println!("dmat shape: {:?}", dmat.shape());
    // specify overall training setup// overall configuration for training/evaluation
    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dmat)                         // dataset to train with
        .boost_rounds(2)                         // number of training iterations
        .booster_params(booster_params)          // model parameters
//        .evaluation_sets(Some(&evaluation_sets)) // optional datasets to evaluate against in each iteration
        .build().unwrap();

    // train booster model, and print evaluation metrics
    println!("\nTraining tree booster...");

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
    let _batches = ctx.sql(sql).await?.collect().await?;

    assert_eq!(_batches[0].column(0).len(), 8124);
    Ok(())
}
