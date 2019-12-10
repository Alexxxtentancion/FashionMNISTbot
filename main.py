import os
import random
from multiprocessing import Pool
from io import BytesIO

import requests
from vk_api import VkApi
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
import tensorflow as tf
import numpy as np
from PIL import Image

from utils.utils import mnist_class_mapping


vk_session = VkApi(token='6f4e109c2e60f330b15de57da8de7e64a3e809ab8ce43d076e48dd92419d26a9a2a46c1928bac6045c21a')
vk = vk_session.get_api()
longpoll = VkBotLongPoll(vk_session, 171810806)

if __name__ == '__main__':
    model = tf.keras.models.load_model('fashion_mnist_dense.h5')
    for event in longpoll.listen():
        if event.type == VkBotEventType.MESSAGE_NEW:
            stream = BytesIO()
            r = requests.get(event.obj.get('attachments')[0].get('photo').get('sizes')[5]['url'])
            img = BytesIO(r.content) if r.status_code == 200 else None
            img = tf.keras.preprocessing.image.load_img(BytesIO(r.content), target_size=(28, 28),
                                                        color_mode='grayscale')
            x = tf.keras.preprocessing.image.img_to_array(img)
            x = x.reshape(1, 784)
            x = 255 - x
            x /= 255
            prediction = model.predict(x)
            vk.messages.send(
                user_id=event.obj.get('from_id'),
                random_id=random.randint(pow(10, 5), pow(10, 6)),
                message=mnist_class_mapping[np.argmax(prediction)]
            )