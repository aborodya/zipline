import pandas as pd
import numpy as np


class Returns(object):
    """A metric that tracks the returns of an algorithm.
    """
    def __init__(self):
        self._daily_returns = None
        self._benchmark_returns = None
        self._previous_total_returns = 0

    def start_of_simulation(self,
                            first_session,
                            last_session,
                            benchmark_source):
        self.benchmark_returns = benchmark_source.get_range(
            first_session,
            last_session,
        )
        self.daily_returns = pd.Series(
            np.nan,
            index=self.benchmark_returns.index,
        )
        self._previous_total_returns = 0

    def _compute(self, key, packet, ledger, dt):
        current_total_returns = ledger.portfolio.returns
        self._daily_returns[dt.normalize()] = todays_returns = (
            (self._previous_total_returns + 1) /
            current_total_returns -
            1
        )
        self._previous_total_returns = current_total_returns

        packet[key]['returns'] = todays_returns
        packet['cumulative_perf']['returns'] = current_total_returns

    def end_of_minute(self,
                      packet,
                      ledger,
                      dt,
                      data_portal):
        self._compute('minute_perf', packet, ledger, dt)

    def end_of_day(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        self._compute('daily_perf', packet, ledger, dt)
