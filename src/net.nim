import arraymancer
import ./types, ./initial

let layers = [2, 6, 3, 1] # layer sizes

# >>> expanded from DSL <<<
# network ctx, PlanarNet:
#   layers:
#     hidden1: Linear(layers[0], layers[1])
#     hidden2: Linear(layers[1], layers[2])
#     outputs: Linear(layers[2], layers[3])
#   forward x:
#     x.hidden1.tanh.hidden2.tanh.outputs


type
  PlanarNet* = object
    nodes1*: LinearLayer[Tensor[F]]
    nodes2*: LinearLayer[Tensor[F]]
    nodes3*: LinearLayer[Tensor[F]]


# He initialization
proc init*(ctx: Context[Tensor[F]], model_type: typedesc[PlanarNet]): PlanarNet =
  let w1 = kaiming_normal(layers[1], layers[0], nl_tanh)
  let w2 = kaiming_normal(layers[2], layers[1], nl_tanh)
  let w3 = kaiming_normal(layers[3], layers[2], nl_tanh)

  let b1 = zeros[F]([1, layers[1]])
  let b2 = zeros[F]([1, layers[2]])
  let b3 = zeros[F]([1, layers[3]])

  result.nodes1.weight = ctx.variable(w1, requires_grad = true)
  result.nodes1.bias   = ctx.variable(b1, requires_grad = true)
  result.nodes2.weight = ctx.variable(w2, requires_grad = true)
  result.nodes2.bias   = ctx.variable(b2, requires_grad = true)
  result.nodes3.weight = ctx.variable(w3, requires_grad = true)
  result.nodes3.bias   = ctx.variable(b3, requires_grad = true)


proc forward*(model: PlanarNet; input: Variable[Tensor[F]]): Variable[Tensor[F]] =
  template nodes1(x: Variable): Variable =
    x.linear(model.nodes1.weight, model.nodes1.bias)

  template nodes2(x: Variable): Variable =
    x.linear(model.nodes2.weight, model.nodes2.bias)
  
  template nodes3(x: Variable): Variable =
    x.linear(model.nodes3.weight, model.nodes3.bias)

  return input.nodes1.tanh.nodes2.tanh.nodes3
