import arraymancer, random, math, sequtils


# arraymancer private
proc randomNormal(mean = 0.0, std = 1.0): float =
  var valid {.global.} = false
  var x {.global.}, y {.global.}, rho {.global.}: float
  if not valid:
    x = rand(1.0)
    y = rand(1.0)
    rho = sqrt(-2.0 * ln(1.0 - y))
    valid = true
    return rho * cos(2.0*PI*x)*std + mean
  else:
    valid = false
    return rho * sin(2.0*PI*x)*std + mean


proc makePlanar(m: int, makedot: proc(i, j, k: int): (float, float) {.closure.}): auto =
  var xs = zeros[float]([m, 2])
  var ys = zeros[float]([m, 1])
  let k = m div 2

  for j in 0..1:
    for i in 0..<k:
      let n = k*j + i
      let dot = makedot(i, j, k)
      xs[n, 0] = dot[0]
      xs[n, 1] = dot[1]
      ys[n, 0] = j.float

  return (xs, ys)


proc makePetals*(m: int, arm=4.0, blur=0.2, pi=3.12): auto =
  var theta {.global.}, radius {.global.}: float
  makePlanar(m) do (i, j, k: int) -> (float, float):
    theta = j.float * pi + i.float / k.float * pi + blur * randomNormal()
    radius = arm * sin(4 * theta) + blur * randomNormal()
    return (radius * sin(theta),
            radius * cos(theta))


proc makeSpirals*(m: int, blur=1.0): auto =
  var theta {.global.}, radius {.global.}: float
  makePlanar(m) do (i, j, k: int) -> (float, float):
    let c = if j==0: 4.0 else: 2.0
    let d = if j==0: 0.3 else: 0.2
    theta = c * PI * (j+1).float * i.float / k.float # + blur * randomNormal()
    radius = d * theta ^ 2 + blur * randomNormal()
    return (radius * sin(theta),
            radius * cos(theta))


proc makeMoons*(m: int, blur=0.15): auto =
  var theta {.global.}: float
  makePlanar(m) do (i, j, k: int) -> (float, float):
    theta = PI * i.float / k.float
    if j==0:
      return (-0.50 + cos(theta) + blur * randomNormal(),
              -0.25 + sin(theta) + blur * randomNormal())
    if j==1:
      return ( 0.50 - cos(theta) + blur * randomNormal(),
               0.25 - sin(theta) + blur * randomNormal())


proc makeCircles*(m: int, inner = 0.7, outer = 1.0, blur=0.1): auto =
  var theta {.global.}, radius {.global.}: float
  makePlanar(m) do (i, j, k: int) -> (float, float):
    theta = 2 * PI * i.float / k.float
    radius = if j==0: inner else: outer
    return (radius * cos(theta) + blur * randomNormal(),
            radius * sin(theta) + blur * randomNormal())


proc planarGrid*(xs: Tensor[float], side: int): auto =
  result = zeros[float]([side * side, 2])
  let lim = xs.data().mapIt(it.abs).max() + 0.1 # NOTE: xs.map.reduce didnt' work

  for x in 0..<side:
    for y in 0..<side:
      let m = x * side + y
      result[m, 0] = (2 * x.float / side.float - 1.0) * lim
      result[m, 1] = (2 * y.float / side.float - 1.0) * lim
