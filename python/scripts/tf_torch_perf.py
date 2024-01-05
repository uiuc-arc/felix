import numpy as np


def get_torch_network(name: str, batch_size: int):
    import torch
    from torchvision.models import mobilenet_v2, resnet50
    from torchvision.models.video import r3d_18
    from tvm.felix import nn

    if name == "resnet50":
        network = resnet50()
        inputs = torch.randn(batch_size, 3, 256, 256)
    elif name == "mobilenet_v2":
        network = mobilenet_v2()
        inputs = torch.randn(batch_size, 3, 256, 256)
    elif name == "r3d_18":
        network = r3d_18()
        inputs = torch.randn(batch_size, 3, 16, 112, 112)
    elif name == "dcgan":
        network, input_size = nn.dcgan()
        inputs = torch.randn(batch_size, *input_size)
    elif name == "vit":
        network, input_size = nn.vit()
        inputs = torch.randn(batch_size, *input_size)
    elif name == "llama_100":
        network, inputs = nn.llama()
    else:
        raise ValueError(f"Invalid network: {name}")
    return network, inputs


def get_tf_network(name, batch_size):
    import tensorflow as tf
    import tf_nns

    if name == "resnet50":
        model = tf.keras.applications.resnet50.ResNet50(weights=None, input_shape=(256, 256, 3))
        return model, tf.random.normal((batch_size, 256, 256, 3))
    elif name == "mobilenet_v2":
        model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            weights=None, input_shape=(256, 256, 3)
        )
        return model, tf.random.normal((batch_size, 256, 256, 3))
    elif name == "r3d_18":
        model = tf_nns.r3d_18()
        return model, tf.random.normal((batch_size, 16, 112, 112, 3))
    elif name == "dcgan":
        model = tf_nns.dcgan()
        return model, tf.random.normal((batch_size, 100))
    elif name == "vit":
        model = tf_nns.vit()
        return model, tf.random.normal((batch_size, 224, 224, 3))
    else:
        raise ValueError(f"Invalid network: {name}")


def measure_torch_lat_us(model, inputs, number: int, repeat: int):
    def _run(model, inputs, number):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(number):
            model(inputs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    assert number >= 1 and repeat >= 1
    model = model.eval().cuda()
    inputs = inputs.cuda()
    _run(model, inputs, 5)
    lats = np.array([_run(model, inputs, number) for _ in range(repeat)])
    return np.array(lats) / number * 1e3


def measure_tf_lat_us(model, inputs, number: int, repeat: int):
    import time

    def _run(model, inputs, number):
        start_time = time.time()
        for _ in range(number):
            model(inputs)
        end_time = time.time()
        return end_time - start_time

    assert number >= 1 and repeat >= 1
    _run(model, inputs, 5)
    lats = np.array([_run(model, inputs, number) for _ in range(repeat)])
    return np.array(lats) / number * 1e6


def main():
    import tensorflow as tf

    # for network in ["resnet50", "mobilenet_v2", "r3d_18", "dcgan", "vit"]:
    #     model, inputs = get_torch_network(network, 1)
    #     model = torch.compile(model, mode="max-autotune")
    #     lats = measure_torch_lat_us(model, inputs, 100, 5)
    #     print(f"PyTorch {network} (batch_size={1}): {lats.mean():.1f} ± {lats.std():.1f} us")
    for network in ["resnet50", "mobilenet_v2", "r3d_18", "dcgan", "vit"]:
        keras_model, inputs = get_tf_network(network, 1)
        tf_model = tf.function(keras_model, jit_compile=True)
        tf_model = tf_model.get_concrete_function(tf.TensorSpec(inputs.shape, inputs.dtype))
        lats = measure_tf_lat_us(tf_model, inputs, 100, 5)
        print(f"TensorFlow {network} (batch_size={1}): {lats.mean():.1f} ± {lats.std():.1f} us")


if __name__ == "__main__":
    main()
