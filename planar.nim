import random, parseopt, strutils
import src/[datasets, plot, network]

var
  show = false
  examples = 400
  epochs = 100
  batch_size = 32
  learning_rate = 1.0

# parse cmdline options
for kind, key, val in getopt():
  case kind
  of cmdLongOption, cmdShortOption:
    case key
    of "plot", "show", "p", "s": show = true
    of "examples", "x":   examples = parseInt(val.string)
    of "epochs", "e":     epochs = parseInt(val.string)
    of "batch_size", "b": batch_size = parseInt(val.string)
    of "rate", "r", "lr": learning_rate = parseFloat(val.string)
    else: discard
  else: discard

randomize()

let (x, y) = makePetals(examples)
if show: showScatter(x, y)

let model = trainModel(x, y, learning_rate, epochs, batch_size)

let grid = planarGrid(x, 40)
let pred = model.predict(grid)
if show: showHeatmap(pred, 40)
