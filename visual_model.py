from models import *
from keras.utils.vis_utils import plot_model

# pip install pydot 
# sudo apt install graphviz

models = []
models.append((model_G(7), 'graphics/models/model_G.png'))
models.append((model_E(8), 'graphics/models/model_E.png'))
models.append((model_4(2), 'graphics/models/model_4.png'))
models.append((model_3(4), 'graphics/models/model_3.png'))

for model, name in models:
    plot_model(model, to_file=name, show_shapes=True, show_layer_names=True, rankdir='TB')

bot_name = 'demo'
model_file = 'model_zoo/' + bot_name + '_bot.yml'
weight_file = 'model_zoo/' + bot_name + '_weights.hd5'

with open(model_file, 'r') as f:
    yml = yaml.load(f)
    pretrained_model = model_from_yaml(yaml.dump(yml))

    # Note that in Keras 1.0 we have to recompile the model explicitly
    pretrained_model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    pretrained_model.load_weights(weight_file)
    # Remove the last 2 layers dense x => 19*19 and activation (softmax)
    # pretrained_model.summary()
    pretrained_model.pop()
    pretrained_model.pop()
    # pretrained_model.summary()

plot_model(pretrained_model, to_file='graphics/models/pretrained_model.png', show_shapes=True, show_layer_names=True, rankdir='TB')

