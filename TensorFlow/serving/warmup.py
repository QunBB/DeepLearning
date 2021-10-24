import os
import tensorflow as tf
from tensorflow_serving.apis import model_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2


def main():
    serving_dir = "serving-model"
    version = "3"

    with tf.python_io.TFRecordWriter(
            os.path.join(serving_dir, version, "assets.extra/tf_serving_warmup_requests")) as writer:
        request = predict_pb2.PredictRequest(
            model_spec=model_pb2.ModelSpec(name="inception", signature_name='serving_default'),
            inputs={"x1": tf.make_tensor_proto([[1.0, 2.0]], shape=[1, 2]),
                    "inputs_id": tf.make_tensor_proto([[1, 2]], shape=[1, 2])}
        )

        log = prediction_log_pb2.PredictionLog(
            predict_log=prediction_log_pb2.PredictLog(request=request))
        writer.write(log.SerializeToString())


if __name__ == "__main__":
    main()
