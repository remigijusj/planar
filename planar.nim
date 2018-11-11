import ./planar_data
import ./planar_plot
import ./planar_nets

let (x, y) = makePetals(400)
# showScatter(x, y)

let model = trainModel(x, y)

let grid = planarGrid(x, 40)
let pred = model.predict(grid)
# showHeatmap(pred, 40)
discard pred
