import arraymancer, strformat, random
import ./types, ./net

let ctx = newContext(Tensor[F]) # autograd/neural network graph


proc optimizer(model: PlanarNet, learning_rate: F): auto =
  return optimizerSGD(model, learning_rate)
  
  
proc predict*(model: PlanarNet, x_test: Tensor[float]): Tensor[float] =
  ctx.no_grad_mode:
    let x_pred = ctx.variable(x_test.astype(F))
    let y_pred = model.forward(x_pred).value.sigmoid
    return y_pred.astype(float)
  

# this should be in datasets?
# synchroneously shuffle planar data examples
proc shuffleExamples[T](x, y: Tensor[T]) =
  var xd = x.toRawSeq()
  var yd = y.toRawSeq()
  let hi = x.shape[0]-1
  for i in countdown(hi, 1):
    let j = rand(i)
    swap(xd[i*2], xd[j*2])
    swap(xd[i*2+1], xd[j*2+1])
    swap(yd[i], yd[j])


# in the process examples get shuffled, see above
proc trainModel*(x_train, y_train: Tensor[float],
                  learning_rate = 1.0,
                  epochs_cnt = 100,
                  batch_size = 32,
                  debug_every = 1
                ): PlanarNet =
  let x = ctx.variable(x_train.astype(F))
  let y = y_train.astype(F)
  let examples = y.shape[0]

  let model = ctx.init(PlanarNet)
  let optim = model.optimizer(learning_rate.F)

  # mini-batch gradient descent
  for epoch in 0..<epochs_cnt:
    shuffleExamples(x.value, y)
    let max_batch = ceil(examples/batch_size).int - 1
    for batch_id in 0..max_batch:
      let offset = batch_id * batch_size
      let limit = min(offset + batch_size, examples)
      let batch  = x[offset ..< limit, _]
      let target = y[offset ..< limit, _]

      let output = model.forward(batch)
      let loss = sigmoid_cross_entropy(output, target)

      loss.backprop()
      optim.update()

      if debug_every > 0 and (epoch mod debug_every == 0) and batch_id == max_batch:
        ctx.no_grad_mode:
          let y_pred = output.value.sigmoid.round
          let score = accuracy_score(target, y_pred)
          echo &"Epoch {epoch:2d} => loss {loss.value[0]:.3f} Score {score:.3f}%"

  return model
