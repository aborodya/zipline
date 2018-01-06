import numpy as np
import pandas as pd

from zipline.utils.exploding_object import ExplodingObject


class SimplePortfolioField(object):
    """Keep a daily record of a field of the
    :class:`~zipline.protocol.Portfolio` object.

    Parameters
    ----------
    portfolio_field : str
        The portfolio field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``portfolio_field`` will be used.
    """
    def __init__(self, portfolio_field, packet_field=None):
        self._portfolio_field = self._packet_field = portfolio_field
        if packet_field is not None:
            self._packet_field = packet_field

    def start_of_simulation(self,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._daily_value = pd.Series(np.nan, index=sessions)

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf'][self._packet_field] = getattr(
            ledger.portfolio,
            self._portfolio_field,
        )

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        value = getattr(
            ledger.portfolio,
            self._portfolio_field,
        )
        packet['daily_perf'][self._packet_field] = value
        self._daily_value[session] = value

    def end_of_simulation(self, packet, ledger):
        packet[self._packet_field] = self._daily_value.tolist()


class Returns(object):
    """Tracks daily and cumulative returns for the algorithm.
    """
    def start_of_simulation(self,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._previous_total_returns = 0

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        current_total_returns = ledger.portfolio.returns
        todays_returns = (
            (self._previous_total_returns + 1) /
            (current_total_returns + 1) -
            1
        )

        packet['minute_perf']['returns'] = todays_returns
        packet['cumulative_perf']['returns'] = current_total_returns

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf']['returns'] = ledger.daily_returns[session]
        packet['cumulative_perf']['returns'] = r = ledger.portfolio.returns
        self._previous_total_returns = r

    def end_of_simulation(self, packet, ledger):
        packet['cumulative_algorithm_returns'] = (
            (1 + ledger.daily_returns).prod() - 1
        )
        packet['daily_algorithm_returns'] = ledger.daily_returns.tolist()


class BenchmarkReturns(object):
    """Tracks daily and cumulative returns for the benchmark.
    """
    def start_of_simulation(self,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._daily_returns = benchmark_source.daily_returns(
            sessions[0],
            sessions[-1],
        )
        self._daily_cumulative_returns = (
            (1 + self._daily_returns).cumprod() - 1
        )
        if emission_rate == 'daily':
            self._minute_returns = ExplodingObject('self._minute_returns')
            self._minute_cumulative_returns = ExplodingObject(
                'self._minute_cumulative_returns',
            )
        else:
            open_ = trading_calendar.market_open(sessions[0])
            close = trading_calendar.market_close(sessions[-1])
            returns = benchmark_source.get_range(open_, close)
            self._minute_returns = returns.groupby(pd.Timegrouper('D')).apply(
                lambda g: (g + 1).cumprod() - 1,
            )
            self._minute_cumulative_returns = (1 + returns).cumprod() - 1

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['benchmark_returns'] = self._minute_returns[dt]
        packet['cumulative_perf']['benchmark_returns'] = (
            self._minute_cumulative_returns[dt]
        )

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf']['benchmark_returns'] = (
            self._daily_returns[session]
        )
        packet['cumulative_perf']['benchmark_returns'] = (
            self._daily_cumulative_returns[session]
        )

    def end_of_simulation(self, packet, ledger):
        packet['cumulative_algorithm_returns'] = (
            self._daily_cumulative_returns[-1]
        )
        packet['daily_algorithm_returns'] = self._daily_returns.tolist()


class PNL(object):
    """Tracks daily and total PNL.
    """
    def start_of_simulation(self,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        # We start the index at -1 because we want to point the previous day.
        # -1 will wrap around and point to the *last* day; however, we
        # initialize the whole series to 0 so this will give us the results
        # we want without an explicit check.
        self._pnl_index = -1
        self._pnl = pd.Series(0, index=sessions)

    def _compute_pnl_in_period(self, ledger):
        return ledger.portfolio.pnl - self._pnl[self._pnl_index]

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['pnl'] = self._compute_pnl_in_period(ledger)
        packet['cumulative_perf']['pnl'] = ledger.portfolio.pnl

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf']['pnl'] = self._compute_pnl_in_period(ledger)
        packet['cumulative_perf']['pnl'] = pnl = ledger.portfolio.pnl
        self._pnl_index += 1
        self._pnl[self._pnl_index] = pnl

    def end_of_simulation(self, packet, ledger):
        packet['total_pnl'] = ledger.portfolio.pnl
        packet['daily_pnl'] = self._pnl.tolist()


def default_metrics():
    """The set of default metrics.
    """
    return {
        Returns(),
        BenchmarkReturns(),
        SimplePortfolioField('positions_exposure', 'ending_exposure'),
        SimplePortfolioField('positions_value', 'ending_value'),
        SimplePortfolioField('portfolio_value'),
        PNL(),
    }
