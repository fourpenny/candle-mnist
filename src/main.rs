use candle_core::{Device, Tensor, DType, Var};
use candle_nn::{Linear, Module, VarBuilder};
use candle_datasets::vision::Dataset;
use hf_hub::{api::sync::Api, Repo, RepoType};
use parquet::file::reader::SerializedFileReader;

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

fn train_loop(
    dataset: candle_datasets::vision::Dataset,
    model: Box<dyn candle_tut::models::Model>
) -> Result<(), Box<dyn std::error::Error>> {
//    See the example: https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs#L174
    let device = Device::cuda_if_available(0)?;

    let train_labels = dataset.train_labels;
    let train_images = dataset.train_images.to_device(&device)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&device)?;
//    Here I'll need to create the optimizer and loss function

//    Then do each epoch of training where I...
//    Create batches from the dataset (shuffle the indexes)
//    Predict, get loss, then step the optim
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use Device::new_cuda(0)?; to use the GPU.
    let device = Device::Cpu;

    // Load the dataset
    let dataset = get_mnist_dataset().unwrap();

    let weight = Tensor::randn(0f32, 1.0, (100, 784), &device)?;
    let bias = Tensor::rand(0f32, 1.0, (100,), &device)?;
    let first = Linear::new(weight, Some(bias));

    let weight = Tensor::randn(0f32, 1.0, (10, 100), &device)?;
    let bias = Tensor::randn(0f32, 1.0, (10,), &device)?;
    let second = Linear::new(weight, Some(bias));
    
    let model = MLP { first, second };

    let dummy_image = Tensor::randn(0f32, 1.0, (1, 784), &device)?;

    let digit = model.forward(&dummy_image)?;
    println!("Digit {digit:?} digit");
    Ok(())
}