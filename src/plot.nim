import arraymancer, plotly, chroma, sequtils

const
  red  = Color(r: 0.9, g: 0.0, b: 0.0, a: 1.0)
  blue = Color(r: 0.0, g: 0.0, b: 0.9, a: 1.0)
  white = Color(r: 0.9, g: 0.9, b: 0.9, a: 1.0)
  black = Color(r: 0.0, g: 0.0, b: 0.0, a: 1.0)


proc plotScatter*(xs, ys: Tensor[float], title="Plot", plotsize=500, dotsize=8.0): Plot[float] =
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

  Plot[float](layout: layout, traces: @[d])


proc plotHeatmap*(data: seq[seq[float]], scale: seq[float], title="Plot", plotsize=500): Plot[float] =
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

  Plot[float](layout: layout, traces: @[d])


proc plotContour*(data: seq[seq[float]], scale: seq[float], title="Plot", plotsize=500): Plot[float] =
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

  Plot[float](layout: layout, traces: @[d])


proc plotLines*(data: seq[float], title="Plot"): Plot[float] =
  let d = Trace[float](mode: PlotMode.Lines, `type`: PlotType.Scatter)

  d.ys = data

  let layout = Layout(
      title: title,
      width: 1200,
      height: 400,
      xaxis: Axis(title: "x-axis"),
      yaxis: Axis(title: "y-axis"),
      autosize: false)

  Plot[float](layout: layout, traces: @[d])


proc plotCombined*(xs, ys: Tensor[float], zs: seq[seq[float]], scale: seq[float], ps: seq[float]): Plot[float] =
  let d1 = Trace[float](mode: PlotMode.Lines, `type`: PlotType.Contour, name: "grid")
  let d2 = Trace[float](mode: PlotMode.Markers, `type`: PlotType.Scatter, name: "dots")
  let d3 = Trace[float](mode: PlotMode.Lines, `type`: PlotType.Scatter, name: "score")

  d1.zs = zs
  d1.xs = scale
  d1.ys = scale
  d1.colorscale = ColorMap.Bluered
  d1.heatmap = true
  d1.contours = (0.0, 1.0, 0.5)
  d1.hoverinfo = "none"
  d1.colorbar = ColorBar(x: 1.0, len: 0.95, thickness: 20)

  let m = ys.shape[0]
  let pts = xs.astype(float)
  d2.xs = pts[_, 0].reshape(m).data()
  d2.ys = pts[_, 1].reshape(m).data()
  d2.hoverinfo = "x+y"

  let dotsizes = @[4.0]
  let red_blue = [black, white]
  let colors = ys.astype(float).data().mapIt(red_blue[it.int])
  d2.marker = Marker[float](size: dotsizes, color: colors)

  d3.ys = ps
  d3.xaxis = "x2"
  d3.yaxis = "y2"
  d3.hoverinfo = "x+y"

  let layout = Layout(
    width: 1200,
    height: 550,
    margin: Margin(left: 80, right: 80, top: 50, bottom: 50),
    hidelegend: true,
    xaxis: Axis(domain: @[0.5, 1.0]),
    yaxis: Axis(domain: @[0.0, 1.0]),
    xaxis2: Axis(domain: @[0.0, 0.46], anchor: "y2"),
    yaxis2: Axis(domain: @[0.0, 1.0], anchor: "x2", range: (0.0, 1.0)))

  Plot[float](layout: layout, traces: @[d1, d2, d3])


proc output*(p: Plot[float], display: bool, file: string): void =
  if display:
    p.show(path=file)
  elif file.len > 0:
    discard p.save(path=file)
