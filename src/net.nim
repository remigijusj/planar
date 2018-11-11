import arraymancer

type F* = float32 # internal datatype for efficiency

let layers = (2, 6, 3, 1) # layer sizes

# >>> expanded from DSL <<<
# network ctx, PlanarNet:
#   layers:
#     hidden1: Linear(layers[0], layers[1])
#     hidden2: Linear(layers[1], layers[2])
#     output:  Linear(layers[2], layers[3])
#   forward x:
#     x.hidden1.tanh.hidden2.tanh.output


type
  PlanarNet* = object
    hidden1*: LinearLayer[Tensor[F]]
    hidden2*: LinearLayer[Tensor[F]]
    output*: LinearLayer[Tensor[F]]


proc init*(ctx: Context[Tensor[F]], model_type: typedesc[PlanarNet]): PlanarNet =
  let w1 = randomTensor([layers[1], layers[0]], max = F(1.0)) .- F(0.5)
  let b1 = randomTensor([1,         layers[1]], max = F(1.0)) .- F(0.5)
  let w2 = randomTensor([layers[2], layers[1]], max = F(1.0)) .- F(0.5)
  let b2 = randomTensor([1,         layers[2]], max = F(1.0)) .- F(0.5)
  let w3 = randomTensor([layers[3], layers[2]], max = F(1.0)) .- F(0.5)
  let b3 = randomTensor([1,         layers[3]], max = F(1.0)) .- F(0.5)
  result.hidden1.weight = ctx.variable(w1, requires_grad = true)
  result.hidden1.bias   = ctx.variable(b1, requires_grad = true)
  result.hidden2.weight = ctx.variable(w2, requires_grad = true)
  result.hidden2.bias   = ctx.variable(b2, requires_grad = true)
  result.output.weight = ctx.variable(w3, requires_grad = true)
  result.output.bias   = ctx.variable(b3, requires_grad = true)


proc forward*(model: PlanarNet; input: Variable[Tensor[F]]): Variable[Tensor[F]] =
  template hidden1(x: Variable): Variable =
    x.linear(model.hidden1.weight, model.hidden1.bias)

  template hidden2(x: Variable): Variable =
    x.linear(model.hidden2.weight, model.hidden2.bias)
  
  template output(x: Variable): Variable =
    x.linear(model.output.weight, model.output.bias)

  return input.hidden1.tanh.hidden2.tanh.output
