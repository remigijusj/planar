import macros

macro buildPlanarNet*(size: static[int], activation: static[string]): untyped =
  result = newStmtList()

  template defNetwork(): untyped =
    network ctx, PlanarNet:
      layers:
        inputs: Linear(2, hyper.layers[0])
      initialize:
        Scheme(uniform, fake)
      forward x:
        x.inputs

  template defLayer(i: untyped) =
    layer: Linear(hyper.layers[i], hyper.layers[i+1])

  template infixLayer(root: untyped) =
    root.fake.layer

  let net = getAst(defNetwork())
  let layers = net[3][0][1]
  let initial = net[3][1][1]
  let forward = net[3][2][2]

  initial[0][2] = ident(activation)
  case activation:
  of "relu":
    initial[0][0] = ident("He")
  of "tanh":
    initial[0][0] = ident("Xavier")
  else:
    initial[0][0] = ident("Simple")

  for i in 0..<size:
    let layer = getAst(defLayer(i))
    layer[0] = ident("layer" & $i)
    layers.add(layer)

    let infix = getAst(infixLayer(forward[0]))
    infix[1] = ident("layer" & $i)
    infix[0][1] = ident(activation)
    forward[0] = infix

  # echo net.repr
  result.add net
