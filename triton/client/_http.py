"""
https://github.com/triton-inference-server/client/tree/main/src/python/examples
"""

import gevent.ssl
import numpy as np
import tritonclient.http as httpclient


def client_init(url="localhost:8000",
                ssl=False, key_file=None, cert_file=None, ca_certs=None, insecure=False,
                verbose=False):
    """

    :param url:
    :param ssl: Enable encrypted link to the server using HTTPS
    :param key_file: File holding client private key
    :param cert_file: File holding client certificate
    :param ca_certs: File holding ca certificate
    :param insecure: Use no peer verification in SSL communications. Use with caution
    :param verbose: Enable verbose output
    :return:
    """
    if ssl:
        ssl_options = {}
        if key_file is not None:
            ssl_options['keyfile'] = key_file
        if cert_file is not None:
            ssl_options['certfile'] = cert_file
        if ca_certs is not None:
            ssl_options['ca_certs'] = ca_certs
        ssl_context_factory = None
        if insecure:
            ssl_context_factory = gevent.ssl._create_unverified_context
        triton_client = httpclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=True,
            ssl_options=ssl_options,
            insecure=insecure,
            ssl_context_factory=ssl_context_factory)
    else:
        triton_client = httpclient.InferenceServerClient(
            url=url, verbose=verbose)

    return triton_client


def infer(triton_client, model_name,
          input0='INPUT0', input1='INPUT1',
          output0='OUTPUT0', output1='OUTPUT1',
          request_compression_algorithm=None,
          response_compression_algorithm=None):
    """

    :param triton_client:
    :param model_name:
    :param input0:
    :param input1:
    :param output0:
    :param output1:
    :param request_compression_algorithm: Optional HTTP compression algorithm to use for the request body on client side.
            Currently supports "deflate", "gzip" and None. By default, no compression is used.
    :param response_compression_algorithm:
    :return:
    """
    inputs = []
    outputs = []
    # batch_size=8
    # 如果batch_size超过配置文件的max_batch_size，infer则会报错
    # INPUT0、INPUT1为配置文件中的输入节点名称
    inputs.append(httpclient.InferInput(input0, [8, 2], "FP32"))
    inputs.append(httpclient.InferInput(input1, [8, 2], "INT32"))

    # Initialize the data
    # np.random.seed(2022)
    inputs[0].set_data_from_numpy(np.random.random([8, 2]).astype(np.float32), binary_data=False)
    # np.random.seed(2022)
    inputs[1].set_data_from_numpy(np.random.randint(0, 20, [8, 2]).astype(np.int32), binary_data=False)

    # OUTPUT0、OUTPUT1为配置文件中的输出节点名称
    outputs.append(httpclient.InferRequestedOutput(output0, binary_data=False))
    outputs.append(httpclient.InferRequestedOutput(output1,
                                                   binary_data=False))
    query_params = {'test_1': 1, 'test_2': 2}
    results = triton_client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs,
        request_compression_algorithm=request_compression_algorithm,
        response_compression_algorithm=response_compression_algorithm)
    print(results)
    # 转化为numpy格式
    print(results.as_numpy(output0))
    print(results.as_numpy(output1))


if __name__ == '__main__':
    import time

    client = client_init()

    s = time.time()

    infer(triton_client=client, model_name='tf_savemodel')
    #
    # infer(triton_client=client, model_name='tf_graphdef')
    #
    # infer(triton_client=client, model_name='torch_model',
    #       input0='INPUT__0', input1='INPUT__1',
    #       output0='OUTPUT__0', output1='OUTPUT__1')
    #
    # infer(triton_client=client, model_name='tf_onnx',
    #       input0='INPUT:0', input1='INPUT:1')

    # infer(triton_client=client, model_name='torch_onnx')

    # infer(triton_client=client, model_name='trt_model')

    print(time.time() - s)
