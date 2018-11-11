import arraymancer, strformat
import ./net

let ctx = newContext(Tensor[F]) # autograd/neural network graph


proc optimizer(model: PlanarNet, learning_rate: F): auto =
  return optimizerSGD(model, learning_rate)
  
  
proc predict*(model: PlanarNet, x_test: Tensor[float]): Tensor[float] =
  ctx.no_grad_mode:
    let x_pred = ctx.variable(x_test.astype(F))
    let y_pred = model.forward(x_pred).value.sigmoid
    return y_pred.astype(float)
  
  
proc trainModel*(x_train, y_train: Tensor[float],
                  learning_rate = 1.0,
                  epochs = 50,
                  batch_size = 32,
                  debug_every = 5
                ): PlanarNet =
  let x = ctx.variable(x_train.astype(F))
  let y = y_train.astype(F)
  let examples = y.shape[0]

  let model = ctx.init(PlanarNet)
  let optim = model.optimizer(learning_rate.F)

  for epoch in 0..<epochs:
    for batch_id in 0..<examples div batch_size: # <<< some at the end may be lost...
      let offset = batch_id * batch_size
      let batch  = x[offset ..< offset + batch_size, _]
      let target = y[offset ..< offset + batch_size, _]

      let output = model.forward(batch)
      let loss = sigmoid_cross_entropy(output, target)

      loss.backprop()
      optim.update()

      if debug_every > 0 and batch_id mod debug_every == 0:
        ctx.no_grad_mode:
          let y_pred = output.value.sigmoid.round
          let score = accuracy_score(target, y_pred)
          echo &"Epoch {epoch:2d} Batch {batch_id:2d} => loss {loss.value[0]:.3f} Score {score:.3f}%"

  return model
  