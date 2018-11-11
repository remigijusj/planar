import random
import ../src/planar

parseOptions()
randomize()

let (x, y) = makePetals(examples)

let (model, scores) = trainModel(x, y, hyper, epochs, debug_every)

let grid = planarGrid(x, grid_size)
let pred = model.predict(grid)

if show:
  showScatter(x, y, "Planar scatter")
  # showHeatmap(pred, grid_size, "Prediction heatmap")
  showContour(pred, grid_size, "Decision boundary")
  # showLines(scores, "Accuracy score")
