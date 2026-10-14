import gpu_ops


def test_transfer_observer_is_scoped_and_records_direction_and_bytes():
    events = []

    with gpu_ops.observe_transfers(lambda direction, size: events.append((direction, size))):
        gpu_ops._record_transfer("cpu_to_gpu", 123)
        gpu_ops._record_transfer("gpu_to_cpu", 456)

    gpu_ops._record_transfer("cpu_to_gpu", 999)
    assert events == [("cpu_to_gpu", 123), ("gpu_to_cpu", 456)]


def test_nested_transfer_observers_restore_parent_scope():
    outer = []
    inner = []

    with gpu_ops.observe_transfers(lambda direction, size: outer.append((direction, size))):
        gpu_ops._record_transfer("cpu_to_gpu", 1)
        with gpu_ops.observe_transfers(
            lambda direction, size: inner.append((direction, size))
        ):
            gpu_ops._record_transfer("gpu_to_cpu", 2)
        gpu_ops._record_transfer("cpu_to_gpu", 3)

    assert outer == [("cpu_to_gpu", 1), ("cpu_to_gpu", 3)]
    assert inner == [("gpu_to_cpu", 2)]


def test_external_provider_boundaries_share_the_same_observer():
    events = []
    with gpu_ops.observe_transfers(
        lambda direction, size: events.append((direction, size))
    ):
        gpu_ops.record_external_transfer("cpu_to_opengl", 4096)
        gpu_ops.record_external_transfer("opengl_to_cpu", 3072)
    assert events == [
        ("cpu_to_opengl", 4096),
        ("opengl_to_cpu", 3072),
    ]
