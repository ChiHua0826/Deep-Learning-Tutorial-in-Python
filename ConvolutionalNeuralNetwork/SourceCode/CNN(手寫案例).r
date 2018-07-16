#�w��tensorflow�禡�w
install.packages('tensorflow')
#�w��keras�禡�w
install.packages('keras')

#�ޥ�tensorflow�禡�w
library(tensorflow)
#�ޥ�keras�禡�w
library(keras)

#���O�ƶq
num_classes <- 10
#�Ϥ����e
img_rows <- 28
img_cols <- 28

#Ū���V�m���
mnist <- dataset_mnist()
X <- mnist$train$x
Y <- mnist$train$y
#�ഫ��matrix��ƫ��A
X <- array_reshape(X, c(nrow(X), img_rows, img_cols, 1)) #�Ϥ��j�p��28x28
Y <- to_categorical(Y, num_classes)

#�]�w�üƺؤl
use_session_with_seed(0)

#�]�w���g�������c
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 1, kernel_size = c(3, 3), #�L�o���j�p��3x3
                activation = 'relu', input_shape = c(img_rows, img_cols, 1)) %>% 
  layer_flatten() %>% 
  layer_dense(units = num_classes, activation = 'softmax')

#�]�w���g�����ǲߥؼ�
model %>% compile(
  loss='categorical_crossentropy', #�B�γ̤p����~�t�p��~�t
  optimizer='sgd', #��פU��
  metrics = c("accuracy")
)

#�V�m���g����
history <- model %>% fit(
  X, #��J�Ѽ�
  Y, #��X�Ѽ�
  epochs = 100, #�V�m�^�X��
  batch_size = 60000 #�C60000���ץ��v��
)

#��ܯ��g�����v����
model$get_weights()

#�N���ո�ƥN�J�ҫ��i��w��,�è��o�w�����G
results <- model %>% predict(
  X
)

#�e�{���p���G
print(results)
