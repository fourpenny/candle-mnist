use candle_nn::{Linear, Module, VarBuilder};
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