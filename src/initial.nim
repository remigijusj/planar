import arraymancer, math
import ./types

type
  Nonlinearity* = enum nl_sigmoid, nl_tanh, nl_relu, nl_leaky_relu
  InitMode*     = enum fan_in, fan_out

proc calculate_gain(nonlinearity: Nonlinearity, neg_slope=0.01): float =
  case nonlinearity
  of nl_sigmoid:
    1.0
  of nl_tanh:
    5.0 / 3.0
  of nl_relu:
    sqrt(2.0)
  of nl_leaky_relu:
    sqrt(2.0 / (1 + neg_slope ^ 2))


proc xavier_normal*(cl, pl: int, nonlinearity=nl_tanh, neg_slope=0.0): Tensor[F] =
  let gain = calculate_gain(nonlinearity, neg_slope)
  let fan_in = pl.float
  let fan_out = cl.float
  let std = gain * sqrt(2.0 / (fan_in + fan_out))
  randomNormalTensor[F]([cl, pl], 0.0, std)


proc kaiming_normal*(cl, pl: int, nonlinearity=nl_tanh, neg_slope=0.0, mode=fan_in): Tensor[F] =
  let gain = calculate_gain(nonlinearity, neg_slope)
  let fan = (if mode == fan_in: pl else: cl).float
  let std = gain / sqrt(fan)
  randomNormalTensor[F]([cl, pl], 0.0, std)
