import arraymancer, strformat, random

randomize()

type
  F = float32 # network internal datatype for efficiency
  T = Tensor[F]
  TwoLayersNet = object
    hidden: LinearLayer[Tensor[F]]
    output: LinearLayer[Tensor[F]]

let
  layers = (2, 6, 1) # layer sizes
  ctx = newContext(T) # autograd/neural network graph


proc initModel(): TwoLayersNet =
  let w1 = randomTensor([layers[1], layers[0]], max = F(1.0)) .- F(0.5)
  let b1 = randomTensor([1,         layers[1]], max = F(1.0)) .- F(0.5)
  let w2 = randomTensor([layers[2], layers[1]], max = F(1.0)) .- F(0.5)
  let b2 = randomTensor([1,         layers[2]], max = F(1.0)) .- F(0.5)
  result.hidden.weight = ctx.variable(w1, requires_grad = true)
  result.hidden.bias   = ctx.variable(b1, requires_grad = true)
  result.output.weight = ctx.variable(w2, requires_grad = true)
  result.output.bias   = ctx.variable(b2, requires_grad = true)


proc forward(model: TwoLayersNet; input: Variable[T]): Variable[T] =
  template hidden(x: Variable): Variable =
    x.linear(model.hidden.weight, model.hidden.bias)

  template output(x: Variable): Variable =
    x.linear(model.output.weight, model.output.bias)

  return input.hidden.tanh.output


proc optimizer(model: TwoLayersNet, learning_rate: F): auto =
  return optimizerSGD(model, learning_rate)


proc predict*(model: TwoLayersNet, x_test: Tensor[float]): Tensor[float] =
  ctx.no_grad_mode:
    let x_pred = ctx.variable(x_test.astype(F))
    let y_pred = model.forward(x_pred).value.sigmoid
    return y_pred.astype(float)


proc trainModel*(x_train, y_train: Tensor[float],
                  learning_rate=1.0,
                  iterations=1000,
                  debug_every=100
                ): TwoLayersNet =
  let x = ctx.variable(x_train.astype(F))
  let y = y_train.astype(F)

  let model = initModel()
  let optim = model.optimizer(learning_rate.F)

  for step in 0..<iterations:
    let output = model.forward(x)
    let loss = sigmoid_cross_entropy(output, y)

    loss.backprop()
    optim.update()

    if debug_every > 0 and step mod debug_every == 0:
      ctx.no_grad_mode:
        let y_pred = output.value.sigmoid.round
        let score = accuracy_score(y, y_pred)
        echo &"Step {step:3d}: loss {loss.value[0]:.3f} accuracy {score:.3f}%"

  return model
