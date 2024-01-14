use candle_nn::{Linear, Module, VarBuilder};

pub trait Model : Sized {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error>;
    
}

struct MLP {
    first: Linear,
    second: Linear,
}

impl Model for MLP {
    fn forward(&self, input: &Tensor) -> Result<Tensor, candle_core::Error> {
        let x = self.first.forward(input)?;
        let x = x.relu()?;
        self.second.forward(&x)
    }

    fn new(vars: VarBuilder) -> Result<Self, candle_core::Error> {

    };
}