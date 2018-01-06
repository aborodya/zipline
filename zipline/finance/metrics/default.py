import operator as op

import numpy as np
import pandas as pd

from zipline.utils.exploding_object import ExplodingObject


class SimpleLedgerField(object):
    """Emit the current value of a ledger field every day.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """
    def __init__(self, ledger_field, packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit('.', 1)[-1]
        else:
            self._packet_field = packet_field

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf'][self._packet_field] = self._get_ledger_field(
            ledger,
        )

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf'][self._packet_field] = self._get_ledger_field(
            ledger,
        )


class DailyLedgerField(object):
    """Keep a daily record of a field of the ledger object.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """
    def __init__(self, ledger_field, packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit('.', 1)[-1]
        else:
            self._packet_field = packet_field

    def start_of_simulation(self,
                            ledger,
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
        packet['minute_perf'][self._packet_field] = self._get_ledger_field(
            ledger,
        )

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        value = self._get_ledger_field(ledger)
        packet['daily_perf'][self._packet_field] = value
        self._daily_value[session] = value

    def end_of_simulation(self, packet, ledger):
        packet[self._packet_field] = self._daily_value.tolist()


class StartOfPeriodLedgerField(object):
    """Keep track of the value of a ledger field at the start of the period.

    Parameters
    ----------
    ledger_field : str
        The ledger field to read.
    packet_field : str, optional
        The name of the field to populate in the packet. If not provided,
        ``ledger_field`` will be used.
    """
    def __init__(self, ledger_field, packet_field=None):
        self._get_ledger_field = op.attrgetter(ledger_field)
        if packet_field is None:
            self._packet_field = ledger_field.rsplit('.', 1)[-1]
        else:
            self._packet_field = packet_field

    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._previous_day = self._get_ledger_field(ledger)

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf'][self._packet_field] = self._previous_day

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        packet['daily_perf'][self._packet_field] = self._previous_day
        self._previous_day = self._get_ledger_field(ledger)


class Returns(object):
    """Tracks daily and cumulative returns for the algorithm.
    """
    def start_of_simulation(self,
                            ledger,
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
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        if not len(sessions):
            # a zero-session algorithm; idk what you want from me here
            empty = pd.Series([np.nan])
            self._daily_returns = empty
            self._daily_cumulative_returns = empty
            self._minute_returns = empty
            self._minute_cumulative_returns = empty
            return

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
            open_ = trading_calendar.session_open(sessions[0])
            close = trading_calendar.session_close(sessions[-1])
            returns = benchmark_source.get_range(open_, close)
            self._minute_returns = returns.groupby(pd.TimeGrouper('D')).apply(
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
            self._daily_cumulative_returns.iloc[-1]
        )
        packet['daily_algorithm_returns'] = self._daily_returns.tolist()


class PNL(object):
    """Tracks daily and total PNL.
    """
    def start_of_simulation(self,
                            ledger,
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


class CashFlow(object):
    """Tracks daily and cumulative cash flow.

    Notes
    -----
    For historical reasons, this field is named 'capital_used' in the packets.
    """
    def start_of_simulation(self,
                            ledger,
                            emission_rate,
                            trading_calendar,
                            sessions,
                            benchmark_source):
        self._previous_cash_flow = 0.0

    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        cash_flow = ledger.portfolio.cash_flow
        packet['minute_perf']['capital_used'] = (
            self._previous_cash_flow - cash_flow
        )
        packet['cumulative_perf']['pnl'] = cash_flow

    def end_of_session(self,
                       packet,
                       ledger,
                       session,
                       data_portal):
        cash_flow = ledger.portfolio.cash_flow
        packet['daily_perf']['capital_used'] = (
            self._previous_cash_flow - cash_flow
        )
        packet['cumulative_perf']['cash_flow'] = cash_flow
        self._previous_cash_flow = cash_flow


class Orders(object):
    """Tracks daily orders.
    """
    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['orders'] = ledger.orders(dt)

    def end_of_session(self,
                       packet,
                       ledger,
                       dt,
                       data_portal):
        packet['daily_perf']['orders'] = ledger.orders()


class Transactions(object):
    """Tracks daily transactions.
    """
    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['transactions'] = ledger.transactions(dt)

    def end_of_session(self,
                       packet,
                       ledger,
                       dt,
                       data_portal):
        packet['daily_perf']['transactions'] = ledger.transactions()


class Positions(object):
    """Tracks daily positions.
    """
    def end_of_bar(self,
                   packet,
                   ledger,
                   dt,
                   data_portal):
        packet['minute_perf']['positions'] = ledger.positions(dt)

    def end_of_session(self,
                       packet,
                       ledger,
                       dt,
                       data_portal):
        packet['daily_perf']['positions'] = ledger.positions()


def default_metrics():
    """The set of default metrics.
    """
    return {
        Returns(),
        BenchmarkReturns(),
        PNL(),
        CashFlow(),
        Orders(),
        Transactions(),

        SimpleLedgerField('positions'),

        StartOfPeriodLedgerField(
            'portfolio.positions_exposure',
            'starting_exposure',
        ),
        DailyLedgerField('portfolio.positions_exposure', 'ending_exposure'),

        StartOfPeriodLedgerField(
            'portfolio.positions_exposure',
            'starting_value'
        ),
        DailyLedgerField('portfolio.positions_value', 'ending_value'),

        StartOfPeriodLedgerField('portfolio.cash', 'starting_cash'),
        DailyLedgerField('portfolio.cash', 'ending_cash'),

        StartOfPeriodLedgerField('portfolio.portfolio_value'),
        DailyLedgerField('portfolio.portfolio_value'),

        DailyLedgerField('position_tracker.stats.longs_count'),
        DailyLedgerField('position_tracker.stats.shorts_count'),
        DailyLedgerField('position_tracker.stats.long_value'),
        DailyLedgerField('position_tracker.stats.short_value'),
        DailyLedgerField('position_tracker.stats.long_exposure'),
        DailyLedgerField('position_tracker.stats.short_exposure'),

        DailyLedgerField('account.gross_leverage'),
        DailyLedgerField('account.net_leverage'),
    }
