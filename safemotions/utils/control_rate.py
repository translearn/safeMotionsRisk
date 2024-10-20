import time
import logging


class ControlRate:
    def __init__(self, control_rate, skip_periods=False, debug_mode=False, last_time=None, busy_wait=False):
        self._control_rate = control_rate
        self._period = 1 / self._control_rate
        self._busy_wait = busy_wait
        if last_time is None:
            self._last_time = time.perf_counter()
        else:
            self._last_time = last_time
        self._skip_periods = skip_periods
        self._debug_mode = debug_mode
        self._sleep_counter = 0
        self._precomputation_time = 0

    def reset(self):
        self._sleep_counter = 0
        self._last_time = time.perf_counter()

    def start_control_phase(self):
        current_time = time.perf_counter()
        self._precomputation_time = current_time - self._last_time
        self._last_time = current_time

    @property
    def precomputation_time(self):
        return self._precomputation_time

    @property
    def reward_maximum_relevant_distance(self):
        return None

    @property
    def control_rate(self):
        return self._control_rate

    @property
    def last_time(self):
        return self._last_time

    def sleep(self):
        current_time = time.perf_counter()
        target_time = self._last_time + self._period
        diff_time = target_time - current_time
        if diff_time > 0.0:
            if self._busy_wait:
                while time.perf_counter() < target_time:
                    pass
            else:
                time.sleep(diff_time)
            self._last_time = self._last_time + self._period
        else:
            if self._skip_periods:
                self._last_time = self._last_time + self._period
            else:
                self._last_time = current_time

        if self._debug_mode:
            logging.warning("%s: Should sleep for %s s, slept for %s s", self._sleep_counter, diff_time,
                            time.perf_counter() - current_time)
            self._sleep_counter = self._sleep_counter + 1
