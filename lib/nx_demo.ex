defmodule NxDemo do
  import Nx.Defn

  @default_defn_compiler EXLA

  @moduledoc """
  Documentation for `NxDemo`.
  """

  def run do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> =
      File.read!("data/train-images-idx3-ubyte.gz") |> :zlib.gunzip()

    heatmap =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows, n_cols})
      |> Nx.to_heatmap()

    images =
      images
      |> Nx.from_binary({:u, 8})
      |> Nx.reshape({n_images, n_rows * n_cols}, names: [:batch, :input])
      |> Nx.divide(255)
      |> Nx.to_batched_list(30)

    <<_::32, n_labels::32, labels::binary>> =
      File.read!("data/train-labels-idx1-ubyte.gz") |> :zlib.gunzip()

    labels =
      labels
      |> Nx.from_binary({:u, 8})
      # One-hot encoding.
      |> Nx.reshape({n_labels, 1}, names: [:batch, :output])
      |> Nx.equal(Nx.tensor(Enum.to_list(0..9)))
      #
      |> Nx.to_batched_list(30)

    # {images, labels, heatmap}

    zip = Enum.zip(images, labels) |> Enum.with_index()

    params =
      for epoch <- 1..5, {{images, labels}, batch} <- zip, reduce: init_params() do
        params ->
          IO.puts("epoch #{epoch}, batch #{batch}")
          update(params, images, labels)
      end
  end

  defn init_params do
    # Random weights between input and hidden layer.
    w1 = Nx.random_normal({784, 128}, 0.0, 0.1, names: [:input, :hidden])
    # Random biases for the hidden layer.
    b1 = Nx.random_normal({128}, 0.0, 0.1, names: [:hidden])
    # Random weights between hidden and output layer.
    w2 = Nx.random_normal({128, 10}, 0.0, 0.1, names: [:hidden, :output])
    # Random biases for the output layer.
    b2 = Nx.random_normal({10}, 0.0, 0.1, names: [:output])

    {w1, b1, w2, b2}
  end

  # Accentuate high values, attenuate low values over output axes.
  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [:output], keep_axes: true)
  end

  defn predict({w1, b1, w2, b2}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> softmax()
  end

  defn loss({w1, b1, w2, b2}, images, labels) do
    preds = predict({w1, b1, w2, b2}, images)
    -Nx.sum(Nx.mean(Nx.log(preds) * labels, axes: [:output]))
  end

  @step_size 0.01

  defn update({w1, b1, w2, b2} = params, images, labels) do
    # Find the rate of change for each param for the loss fn.
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad(params, loss(params, images, labels))

    # Adjust each param by rate of change * step size.
    {w1 - grad_w1 * @step_size, b1 - grad_b1 * @step_size, w2 - grad_w2 * @step_size,
     b2 - grad_b2 * @step_size}
  end
end
