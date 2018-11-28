import random
import ../src/planar

parseOptions()

if help: justShowUsage()

echo "Seed: " & $random_seed
randomize(random_seed)

let (x, y) = makePattern(examples, pattern)

let (model, scores) = trainModel(x, y, hyper, epochs, debug_every)

let (grid, scale) = planarGrid(x.limit, grid_step)
let pred = model.predict(grid)

let p = plotCombined(x, y, pred.asGrid, scale, scores)
output(p, display, file)
