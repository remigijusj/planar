import arraymancer, plotly, chroma, sequtils

const
  red  = Color(r: 0.9, g: 0.0, b: 0.0, a: 1.0)
  blue = Color(r: 0.0, g: 0.0, b: 0.9, a: 1.0)


proc showScatter*(xs, ys: Tensor[float], plotsize=500, dotsize=8.0): void =
  let d = Trace[float](mode: PlotMode.Markers, `type`: PlotType.Scatter)
  let m = ys.shape[0]

  let pts = xs.astype(float)
  d.xs = pts[_, 0].reshape(m).data()
  d.ys = pts[_, 1].reshape(m).data()

  let dotsizes = @[dotsize]
  let red_blue = [blue, red]
  let colors = ys.astype(float).data().mapIt(red_blue[it.int])
  d.marker = Marker[float](size: dotsizes, color: colors)

  let layout = Layout(
    title: "Planar scatter",
    width: plotsize,
    height: plotsize,
    autosize: false,
    xaxis: Axis(title: "x-axis"),
    yaxis: Axis(title: "y-axis")
  )

  let p = Plot[float](layout: layout, traces: @[d])
  p.show()


# the data needs to be supplied as a nested seq
proc reshapeData(data: Tensor[float], side: int): seq[seq[float]] =
  result = newSeqWith(side, newSeq[float](side))
  for x in 0..<side:
    for y in 0..<side:
      let m = x * side + y
      result[y][x] = data[m, 0]  # <<< opposite?


proc showHeatmap*(data: Tensor[float], side: int, plotsize=500): void =
  let d = Trace[float](mode: PlotMode.Lines, `type`: PlotType.HeatMap)

  d.zs = reshapeData(data, side)
  d.colormap = ColorMap.Bluered

  let layout = Layout(
    title: "Prediction heatmap",
    width: plotsize,
    height: plotsize,
    autosize: true,
    xaxis: Axis(title: "x-axis"),
    yaxis: Axis(title: "y-axis")
  )

  let p = Plot[float](layout: layout, traces: @[d])
  p.show()
