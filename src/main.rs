use candle_core::{Device, Tensor, DType, D};
use candle_nn::{VarBuilder, VarMap, Optimizer, loss, ops};
use candle_datasets::vision::Dataset;
use hf_hub::{api::sync::Api, Repo, RepoType};
use parquet::file::reader::SerializedFileReader;
use rand::prelude::*;

fn get_mnist_dataset() -> Result<Dataset, Box<dyn std::error::Error>> {
    let dataset_id = "mnist".to_string();

    let api = Api::new()?;

    let repo = Repo::with_revision(
        dataset_id,
        RepoType::Dataset,
        "refs/convert/parquet".to_string(),
    );
    let repo = api.repo(repo);

    let test_parquet_filename = repo.get("mnist/test/0000.parquet")?;
    let train_parquet_filename = repo.get("mnist/train/0000.parquet")?;
    let test_parquet = SerializedFileReader::new(std::fs::File::open(test_parquet_filename)?)?;
    let train_parquet = SerializedFileReader::new(std::fs::File::open(train_parquet_filename)?)?;

    let test_samples = 10_000;
    let mut test_buffer_images: Vec<u8> = Vec::with_capacity(test_samples * 784);
    let mut test_buffer_labels: Vec<u8> = Vec::with_capacity(test_samples);
    for row in test_parquet{
        for (_name, field) in row.unwrap().get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        test_buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                test_buffer_labels.push(*label as u8);
            }
        }
    }
    let test_images = (Tensor::from_vec(test_buffer_images, (test_samples, 784), &Device::Cpu)?.to_dtype(DType::F32)? / 255.)?;
    let test_labels = Tensor::from_vec(test_buffer_labels, (test_samples, ), &Device::Cpu)?;

    let train_samples = 60_000;
    let mut train_buffer_images: Vec<u8> = Vec::with_capacity(train_samples * 784);
    let mut train_buffer_labels: Vec<u8> = Vec::with_capacity(train_samples);
    for row in train_parquet{
        for (_name, field) in row?.get_column_iter() {
            if let parquet::record::Field::Group(subrow) = field {
                for (_name, field) in subrow.get_column_iter() {
                    if let parquet::record::Field::Bytes(value) = field {
                        let image = image::load_from_memory(value.data()).unwrap();
                        train_buffer_images.extend(image.to_luma8().as_raw());
                    }
                }
            } else if let parquet::record::Field::Long(label) = field {
                train_buffer_labels.push(*label as u8);
            }
        }
    }
    let train_images = (Tensor::from_vec(train_buffer_images, (train_samples, 784), &Device::Cpu)?.to_dtype(DType::F32)? / 255.)?;
    let train_labels = Tensor::from_vec(train_buffer_labels, (train_samples, ), &Device::Cpu)?;

    let mnist = candle_datasets::vision::Dataset {
        train_images,
        train_labels,
        test_images,
        test_labels,
        labels: 10,
    };
    return Ok(mnist);
}

//  Train the input type of neural network to classify the given dataset.
//  We also test the accuracy of the network at the end of each training epoch.
fn train_loop<M: candle_tut::models::Model>(
    dataset: candle_datasets::vision::Dataset,
    config: candle_tut::models::Config,
) -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::cuda_if_available(0)?;

    let bsize: usize = config.get_batch_size();

    let train_labels = dataset.train_labels;
    let train_images = dataset.train_images.to_device(&device)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&device)?;

    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = M::new(vs.clone())?;

    let mut sgd = candle_nn::optim::SGD::new(varmap.all_vars(), config.get_lr() )?;
    let test_images = dataset.test_images.to_device(&device)?;
    let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&device)?;

    let n_batches = train_images.dim(0)? / bsize;
    let mut batch_idxs = (0..n_batches).collect::<Vec<usize>>();

    for epoch in 1..config.get_num_epochs() {
        batch_idxs.shuffle(&mut thread_rng());
        let mut sum_loss = 0f32;

        for batch_idx in batch_idxs.iter() {
            let train_images = train_images.narrow(0, batch_idx * bsize, bsize)?;
            let train_labels = train_labels.narrow(0, batch_idx * bsize, bsize)?;
            let logits = model.forward(&train_images, &config)?;
            let log_sm = ops::log_softmax(&logits, D::Minus1)?;
            let loss = loss::nll(&log_sm, &train_labels)?;
            sgd.backward_step(&loss)?;
            sum_loss += loss.to_vec0::<f32>()?;
        }

        let avg_loss = sum_loss / n_batches as f32;

        let test_logits = model.forward(&test_images, &config)?;
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        let test_accuracy = sum_ok / test_labels.dims1()? as f32;
        println!(
            "{epoch:4} train loss {:8.5} test acc: {:5.2}%",
            avg_loss,
            100. * test_accuracy
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the dataset
    let dataset = get_mnist_dataset().unwrap();

    let config = candle_tut::models::Config::new(
        0.05,
        None,
        None,
        10,
        true,
        64
    );

    train_loop::<candle_tut::models::CNN>(dataset, config).unwrap();

    Ok(())
}