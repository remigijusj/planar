import macros

# TODO: macro, use args
template buildPlanarNet*(size: int, activation: untyped): untyped {.dirty.} =
  network ctx, PlanarNet:
    layers:
      hidden0: Linear(2, hyper.layers[0])
      hidden1: Linear(hyper.layers[0], hyper.layers[1])
      outputs: Linear(hyper.layers[1], 1)
    initialize:
      Xavier(uniform, tanh)
    forward x:
      x.hidden0.tanh.hidden1.tanh.outputs
