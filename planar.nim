import random, parseopt, strutils
import src/[datasets, plot, network]

var
  show = true # false
  grid_size = 40
  debug_every = 10
  examples = 500
  epochs = 50 # 100
  batch_size = 32
  learning_rate = 1.0
  beta1 = 0.9
  beta2 = 0.99

# parse cmdline options
for kind, key, val in getopt():
  case kind
  of cmdLongOption, cmdShortOption:
    case key
    of "plot", "show", "p", "s": show = true
    of "grid", "gs":       grid_size = parseInt(val.string)
    of "examples", "x":    examples = parseInt(val.string)
    of "epochs", "e":      epochs = parseInt(val.string)
    of "batch_size", "bs": batch_size = parseInt(val.string)
    of "rate", "r", "lr":  learning_rate = parseFloat(val.string)
    of "beta1", "b1":      beta1 = parseFloat(val.string)
    of "beta2", "b2":      beta2 = parseFloat(val.string)
    else: discard
  else: discard

randomize()

let (x, y) = makePetals(examples)

let (model, scores) = trainModel(x, y, learning_rate, beta1, beta2, epochs, batch_size, debug_every)

let grid = planarGrid(x, grid_size)
let pred = model.predict(grid)

if show:
  showScatter(x, y, "Planar scatter")
  # showHeatmap(pred, grid_size, "Prediction heatmap")
  showContour(pred, grid_size, "Decision boundary")
  # showLines(scores, "Accuracy score")
