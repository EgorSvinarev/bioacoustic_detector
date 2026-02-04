import tensorflow_hub as hub
import tensorflow as tf

yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
class_map = yamnet.class_map_path().numpy().decode("utf-8")

def _load_classes():
    classes = []
    with open(class_map) as f:
        next(f)
        for line in f:
            classes.append(line.split(",")[2])
    return classes

CLASS_NAMES = _load_classes()

def run_inference(audio):
    scores, _, _ = yamnet(audio)
    return tf.reduce_mean(scores, axis=0).numpy()
