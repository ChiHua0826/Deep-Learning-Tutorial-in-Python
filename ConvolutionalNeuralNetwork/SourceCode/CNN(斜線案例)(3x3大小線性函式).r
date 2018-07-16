#�w��tensorflow�禡�w
install.packages('tensorflow')
#�w��keras�禡�w
install.packages('keras')

#�ޥ�tensorflow�禡�w
library(tensorflow)
#�ޥ�keras�禡�w
library(keras)

#Ū���V�m���
X <- read.csv('../��ƶ�/X_�׽u�ר�(3x3).csv', header = FALSE)
Y <- read.csv('../��ƶ�/Y_�׽u�ר�(3x3).csv', header = FALSE)
#�ഫ��matrix��ƫ��A
X <- data.matrix(X)
X <- array_reshape(X, c(nrow(X), 3, 3, 1))
Y <- data.matrix(Y)

#�]�w�üƺؤl
use_session_with_seed(2)

#�]�w���g�������c
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 1, kernel_size = c(2, 2), 
                activation = 'linear', input_shape = c(3, 3, 1)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 1, activation = 'linear')

#�]�w���g�����ǲߥؼ�
model %>% compile(
  loss='mean_squared_error', #�B�γ̤p����~�t�p��~�t
  optimizer='sgd' #��פU��
)

#�V�m���g����
history <- model %>% fit(
  X, #��J�Ѽ�
  Y, #��X�Ѽ�
  epochs = 1000, #�V�m�^�X��
  batch_size = 1 #�v���ץ��v��
)

#��ܯ��g�����v����
model$get_weights()

#�N���ո�ƥN�J�ҫ��i��w��,�è��o�w�����G
results <- model %>% predict(
  X
)

#�e�{���p���G
print(results)
