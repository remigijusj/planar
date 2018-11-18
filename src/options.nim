import parseopt, sequtils, strutils

type
  Hyperparams* = ref object
    layers*: seq[int]
    batch_size*: int
    learning_rate*: float
    beta1*: float
    beta2*: float

proc defaultHyper(): auto =
  Hyperparams(
    layers: @[5, 5, 2, 1],
    batch_size: 32,
    learning_rate: 1.0,
    beta1: 0.9,
    beta2: 0.999
  )

const
  depth* = 3 # >= 1
  activation* = "tanh" # tanh, relu, sigmoid

var
  help* = false
  display* = false
  file* = ""
  pattern* = "petals"
  examples* = 500
  epochs* = 100
  debug_every* = 10
  grid_step* = 0.2
  hyper* = defaultHyper()


proc parseIntsList(val: string): seq[int] =
  result = val.split(',').mapIt(it.parseInt)
  result.add(1)
  assert(result.len == depth + 1)


proc justShowUsage*() =
  echo """
Usage: plan [options]
  --help, -h    Display help
  --show, -s    Display results in browser
  --file, -f    File name to save results
  --pattern,  -p   Pattern of dots (petals|spirals|moons|circles)
  --examples, -x   Number of example dots (500)
  --epochs,   -e   Number of epochs (100)
  --debug,  -d     Debug info every d epochs (10)
  --grid, -gs      Grid step (0.2)
  --layers, -l        Hidden layer sizes (5,5,2,1)
  --batch,  -bs       Batch size (32)
  --rate,   -lr       Learning rate (1.0)
  --beta1,  -b1       Beta1 for Adam (0.9)
  --beta2,  -b2       Beta2 for Adam (0.999)
"""
  quit()

proc parseOptions*(): auto =
  for kind, key, val in getopt():
    case kind
    of cmdLongOption, cmdShortOption:
      case key
      of "help", "h":        help = true
      of "show", "s":        display = true
      of "file", "f":        file = val.string
      of "pattern",  "p":    pattern = val.string
      of "examples", "x":    examples = parseInt(val.string)
      of "epochs",   "e":    epochs = parseInt(val.string)
      of "debug",    "d":    debug_every = parseInt(val.string)
      of "grid",     "gs":   grid_step = parseFloat(val.string)
      of "layers", "l":      hyper.layers = parseIntsList(val.string)
      of "batch",  "bs":     hyper.batch_size = parseInt(val.string)
      of "rate",   "lr":     hyper.learning_rate = parseFloat(val.string)
      of "beta1",  "b1":     hyper.beta1 = parseFloat(val.string)
      of "beta2",  "b2":     hyper.beta2 = parseFloat(val.string)
      else: discard
    else: discard
