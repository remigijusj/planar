import arraymancer, strformat, sequtils, random
import ./options, ./network

type F = float32 # internal datatype for efficiency

let ctx = newContext(Tensor[F])

buildPlanarNet(depth, activation)


proc predict*(model: PlanarNet, x_test: Tensor[float]): Tensor[float] =
  ctx.no_grad_mode:
    let x_pred = ctx.variable(x_test.astype(F))
    let y_pred = model.forward(x_pred).value.sigmoid
    return y_pred.astype(float)


# this should be in datasets?
# synchronously shuffle planar data examples
proc shuffleExamples[T](x, y: Tensor[T]) =
  var xd = x.toRawSeq()
  var yd = y.toRawSeq()
  let hi = x.shape[0]-1
  for i in countdown(hi, 1):
    let j = rand(i)
    swap(xd[i*2], xd[j*2])
    swap(xd[i*2+1], xd[j*2+1])
    swap(yd[i], yd[j])


proc debugHeader(debug_every: int) =
  if debug_every > 0:
    echo "| epo |  loss |   err | maxp |"
    echo "+-----+-------+-------+------+"


proc debugEpoch(debug_every, epoch: int, loss, error, max_p: float) =
  if debug_every > 0 and (epoch mod debug_every == 0):
    echo &"| {epoch:3d} | {loss:.3f} | {error:.3f} | {max_p:.2f} |"


proc makeOptimizer(model: PlanarNet, hyper: Hyperparams): Optimizer[Tensor[F]] =
  if options.optimizer == "adam":
    result = model.optimizer(Adam, hyper.learning_rate.F, hyper.beta1.F, hyper.beta2.F, hyper.epsilon.F, hyper.weight_decay.F)
  else:
    result = model.optimizer(SGD, hyper.learning_rate.F, hyper.weight_decay.F)


# in the process examples get shuffled, see above
proc trainModel*(x_train, y_train: Tensor[float],
                 hyper: Hyperparams,
                 epochs_cnt, debug_every: int
                ): tuple[model: PlanarNet, progress: seq[float]] =
  let x = ctx.variable(x_train.astype(F))
  let y = y_train.astype(F)
  let examples = y.shape[0]
  let max_batch = ceil(examples/hyper.batch_size).int - 1

  let model = ctx.init(PlanarNet)
  let optim = makeOptimizer(model, hyper)
  var progress = newSeqOfCap[float](epochs_cnt)

  debugHeader(debug_every)

  # mini-batch gradient descent
  for epoch in 0..<epochs_cnt:
    shuffleExamples(x.value, y)
    for batch_id in 0..max_batch:
      let offset = batch_id * hyper.batch_size
      let limit = min(offset + hyper.batch_size, examples)
      let batch  = x[offset ..< limit, _]
      let target = y[offset ..< limit, _]

      let output = model.forward(batch)
      let loss = sigmoid_cross_entropy(output, target)

      loss.backprop()
      optim.update()

      if batch_id == max_batch:
        ctx.no_grad_mode:
          let y_pred = output.value.sigmoid.round
          let error = 1.0 - accuracy_score(target, y_pred)
          let max_p = optim.params.map(proc(p: Variable[Tensor[F]]): float = p.value.reduce_inline: x = max(x, y.float.abs))
          progress.add(error)
          debugEpoch(debug_every, epoch, loss.value[0], error, max_p.max)

  return (model, progress)
