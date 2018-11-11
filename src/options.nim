import parseopt, strutils

type
  Hyperparams* = ref object
    batch_size*: int
    learning_rate*: float
    beta1*: float
    beta2*: float

proc defaultHyper(): auto =
  Hyperparams(
    batch_size: 32,
    learning_rate: 1.0,
    beta1: 0.9,
    beta2: 0.99
  )

var
  examples* = 500
  epochs* = 100
  debug_every* = 10
  grid_size* = 40
  show* = false
  hyper* = defaultHyper()


proc parseOptions*(): auto =
  for kind, key, val in getopt():
    case kind
    of cmdLongOption, cmdShortOption:
      case key
      of "plot", "show", "p", "s": show = true
      of "grid", "gs":       grid_size = parseInt(val.string)
      of "examples", "x":    examples = parseInt(val.string)
      of "epochs", "e":      epochs = parseInt(val.string)
      of "batch_size", "bs": hyper.batch_size = parseInt(val.string)
      of "rate", "r", "lr":  hyper.learning_rate = parseFloat(val.string)
      of "beta1", "b1":      hyper.beta1 = parseFloat(val.string)
      of "beta2", "b2":      hyper.beta2 = parseFloat(val.string)
      else: discard
    else: discard
