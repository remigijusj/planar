import arraymancer, math, sequtils


# The range of both x and y is -limit..limit
# returns tuple[grid: Tensor[float], scale: seq[float]]
proc planarGrid*(limit, step: float, places=2): auto =
  let side = (limit/step).int
  let size = 2 * side + 1
  var grid = zeros[float]([size * size, 2])
  var scale = newSeq[float](size)

  for x in 0..<size:
    scale[x] = ((x - side).float * step).round(places)
    for y in 0..<size:
      let n = x * size + y
      grid[n, 0] = scale[x]
      grid[n, 1] = ((y - side).float * step).round(places)

  return (grid, scale)


# NOTE: xs.map.reduce didnt' work
proc limit*(xs: Tensor[float], padding=0.1): float =
  xs.data().mapIt(it.abs).max() + padding


proc asGrid*(data: Tensor[float]): seq[seq[float]] =
  let size = sqrt(data.shape[0].float).int
  result = newSeqWith(size, newSeq[float](size))
  for x in 0..<size:
    for y in 0..<size:
      let n = x * size + y
      result[y][x] = data[n, 0] # Why not opposite?
