import parseopt, strutils

var
  show* = true # false
  grid_size* = 40
  debug_every* = 10
  examples* = 500
  epochs* = 100
  batch_size* = 32
  learning_rate* = 1.0
  beta1* = 0.9
  beta2* = 0.99


proc parseOptions*(): auto =
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
