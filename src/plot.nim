import arraymancer, plotly, chroma, sequtils

const
  red  = Color(r: 0.9, g: 0.0, b: 0.0, a: 1.0)
  blue = Color(r: 0.0, g: 0.0, b: 0.9, a: 1.0)


proc showScatter*(xs, ys: Tensor[float], title="Plot", plotsize=500, dotsize=8.0): void =
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
    title: title,
    width: plotsize,
    height: plotsize,
    autosize: false,
    xaxis: Axis(title: "x-axis"),
    yaxis: Axis(title: "y-axis")
  )

  let p = Plot[float](layout: layout, traces: @[d])
  p.show()


proc showHeatmap*(data: seq[seq[float]], scale: seq[float], title="Plot", plotsize=500): void =
  let d = Trace[float](mode: PlotMode.Lines, `type`: PlotType.HeatMap)

  d.zs = data
  d.xs = scale
  d.ys = scale
  d.colormap = ColorMap.Bluered

  let layout = Layout(
    title: title,
    width: plotsize,
    height: plotsize,
    xaxis: Axis(title: "x-axis"),
    yaxis: Axis(title: "y-axis"),
    autosize: true)

  let p = Plot[float](layout: layout, traces: @[d])
  p.show()


proc showContour*(data: seq[seq[float]], scale: seq[float], title="Plot", plotsize=500): void =
  let d = Trace[float](mode: PlotMode.Lines, `type`: PlotType.Contour)

  d.zs = data
  d.xs = scale
  d.ys = scale
  d.colorscale = ColorMap.Bluered
  d.heatmap = true
  d.contours = (0.0, 1.0, 0.5)

  let layout = Layout(
    title: title,
    width: plotsize,
    height: plotsize,
    xaxis: Axis(title: "x-axis"),
    yaxis: Axis(title: "y-axis"),
    autosize: true)

  let p = Plot[float](layout: layout, traces: @[d])
  p.show()


proc showLines*(data: seq[float], title="Plot"): void =
  let d = Trace[float](mode: PlotMode.Lines, `type`: PlotType.Scatter)

  d.ys = data

  let layout = Layout(
      title: title,
      width: 1200,
      height: 400,
      xaxis: Axis(title: "x-axis"),
      yaxis: Axis(title: "y-axis"),
      autosize: false)

  let p = Plot[float](layout: layout, traces: @[d])
  p.show()
