from emolight.runtime.result_queue import LatestOnlyQueue


def test_result_queue_is_bounded_and_returns_only_latest_value():
    result_queue = LatestOnlyQueue[int]()

    result_queue.publish(1)
    result_queue.publish(2)
    result_queue.publish(3)

    assert result_queue.capacity == 1
    assert result_queue.take_latest() == 3
    assert result_queue.take_latest() is None
    assert result_queue.replaced_count == 2
