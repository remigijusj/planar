import parseopt, sequtils, strutils

type
  Hyperparams* = ref object
    layers*: seq[int]
    batch_size*: int
    learning_rate*: float
    weight_decay*: float
    beta1*: float
    beta2*: float
    epsilon*: float

proc defaultHyper(): auto =
  Hyperparams(
    layers: @[5, 2, 1],
    batch_size: 32,
    learning_rate: 1.0,
    weight_decay: 0.0,
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8
  )

const
  depth* = 2 # >= 1
  activation* = "tanh" # tanh, relu, sigmoid

var
  help* = false
  display* = false
  file* = ""
  pattern* = "petals"
  examples* = 500
  epochs* = 100
  optimizer* = "sgd"
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
  --optim,    -o   Optimizer (sgd|adam)
  --debug,    -d   Debug info every d epochs (10)
  --grid,     -g   Grid step (0.2)
  --layers,   -l      Hidden layer sizes (5,2,1)
  --batch,    -b      Batch size (32)
  --rate,     -r      Learning rate (1.0)
  --wdecay,   -w      Weight decay (0.0)
  --beta1,    -B1     Beta1 for Adam (0.9)
  --beta2,    -B2     Beta2 for Adam (0.999)
  --epsilon,  -E      Epsilon for Adam (1e-8)
"""
  quit()

proc parseOptions*(): auto =
  for kind, key, val in getopt():
    case kind
    of cmdLongOption, cmdShortOption:
      case key
      of "help",     "h":    help = true
      of "show",     "s":    display = true
      of "file",     "f":    file = val.string
      of "pattern",  "p":    pattern = val.string
      of "examples", "x":    examples = parseInt(val.string)
      of "epochs",   "e":    epochs = parseInt(val.string)
      of "optim",    "o":    optimizer = val.string
      of "debug",    "d":    debug_every = parseInt(val.string)
      of "grid",     "g":    grid_step = parseFloat(val.string)
      of "layers",   "l":    hyper.layers = parseIntsList(val.string)
      of "batch",    "b":    hyper.batch_size = parseInt(val.string)
      of "rate",     "r":    hyper.learning_rate = parseFloat(val.string)
      of "wdecay",   "w":    hyper.weight_decay = parseFloat(val.string)
      of "beta1",    "B1":   hyper.beta1 = parseFloat(val.string)
      of "beta2",    "B2":   hyper.beta2 = parseFloat(val.string)
      of "epsilon",  "E":    hyper.epsilon = parseFloat(val.string)
      else: discard
    else: discard
