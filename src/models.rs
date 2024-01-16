use candle_nn::{
    Linear, Module, VarBuilder, Conv2d, ModuleT};
use candle_core::{Tensor, Error};

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

pub trait Model : Sized {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error>;
    // TODO: Make this accept a config
    fn new(vars: VarBuilder) -> Result<Self, Error>;
}

pub struct MLP {
    first: Linear,
    second: Linear,
}

impl Model for MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let x = self.first.forward(input)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }

    fn new(vars: VarBuilder) -> Result<Self, Error> {
        // We modify the default names of the weights/biases to include the layer name ("ln1/2")
        let first = candle_nn::linear(IMAGE_DIM, 100, vars.pp("ln1"))?;
        let second = candle_nn::linear(100, LABELS, vars.pp("ln2"))?;
        Ok(Self { first, second })
    }
}

pub struct CNN {
    conv1: Conv2d,
    conv2: Conv2d,
    fc1: Linear,
    fc2: Linear,
    dropout: candle_nn::Dropout,
}

impl Model for CNN {
    fn new(vs: VarBuilder) -> Result<Self, Error> {
        let conv1 = candle_nn::conv2d(1, 32, 5, Default::default(), vs.pp("c1"))?;
        let conv2 = candle_nn::conv2d(32, 64, 5, Default::default(), vs.pp("c2"))?;
        let fc1 = candle_nn::linear(1024, 1024, vs.pp("fc1"))?;
        let fc2 = candle_nn::linear(1024, LABELS, vs.pp("fc2"))?;
        let dropout = candle_nn::Dropout::new(0.5);
        Ok(Self {
            conv1,
            conv2,
            fc1,
            fc2,
            dropout,
        })
    }
    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        // Get the batch and image dimensions from the tensor
        let (b_sz, _img_dim) = xs.dims2()?;
        let train: bool = true;
        let xs = xs
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .flatten_from(1)?
            .apply(&self.fc1)?
            .relu()?;
        self.dropout.forward_t(&xs, train)?.apply(&self.fc2)
    }
}