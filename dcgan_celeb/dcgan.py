def celeb():
  X = util.get_celeb()
  # Como temos muitas imagens, vamos carregar apenas os endereços e deixar para carregar cada imagem durante o treinamento
  dim = 64
  colors = 3

  # aqui definiremos a estrutura da rede
  d_sizes = {
    'conv_layers': [
      (64, 5, 2, False),
      (128, 5, 2, True),
      (256, 5, 2, True),
      (512, 5, 2, True)
    ],
    'dense_layers': [],
  }
  g_sizes = {
    'z': 100,
    'projection': 512,
    'bn_after_project': True,
    'conv_layers': [
      (256, 5, 2, True),
      (128, 5, 2, True),
      (64, 5, 2, True),
      (colors, 5, 2, False)
    ],
    'dense_layers': [],
    'output_activation': tf.tanh,
  }

  # setup gan
  # note: assume square images, so only need 1 dim
  # gan = DCGAN(dim, colors, d_sizes, g_sizes)
  # gan.fit(X)