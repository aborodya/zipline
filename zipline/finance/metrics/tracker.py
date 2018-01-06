#
# Copyright 2017 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import division

import logbook

from zipline.errors import NoFurtherDataError
from ..ledger import Ledger


log = logbook.Logger(__name__)


class MetricsTracker(object):
    """The algorithm's interface to the registered risk and performance
    metrics.

    Parameters
    ----------
    trading_calendar : TrandingCalendar
        The trading calendar used in the simulation.
    first_session : pd.Timestamp
        The label of the first trading session in the simulation.
    last_session : pd.Timestamp
        The label of the last trading session in the simulation.
    capital_base : float
        The starting capital for the simulation.
    emission_rate : {'daily', 'minute'}
        How frequently should a performance packet be generated?
    asset_finder : AssetFinder
        The asset finder used in the simulation.
    adjustment_reader : AdjustmentReader
        The adjustment reader used in the simulation.
    benchmark_source : BenchmarkSource
        The benchmark return source for the given simulation..
    metrics : list[Metric]
        The metrics to track.
    """
    _hooks = (
        'start_of_simulation',
        'end_of_bar',
        'end_of_session',
        'end_of_simulation',
    )

    def __init__(self,
                 trading_calendar,
                 first_session,
                 last_session,
                 capital_base,
                 emission_rate,
                 asset_finder,
                 adjustment_reader,
                 benchmark_source,
                 metrics):
        self.emission_rate = emission_rate

        self._trading_calendar = trading_calendar
        self._first_session = first_session
        self._last_session = last_session
        self._capital_base = capital_base
        self._asset_finder = asset_finder
        self._adjustment_reader = adjustment_reader

        self._current_session = first_session
        (self._market_open,
         self._market_close) = trading_calendar.open_and_close_for_session(
             first_session,
        )
        self._session_count = 0

        sessions = trading_calendar.sessions_in_range(
            first_session,
            last_session,
        )
        self._total_session_count = len(sessions)

        self._ledger = Ledger(sessions, capital_base, emission_rate)

        # bind all of the hooks from the passed metric objects.
        for hook in self._hooks:
            registered = []
            for metric in metrics:
                try:
                    registered.append(getattr(metric, hook))
                except AttributeError:
                    pass

            def closing_over_loop_variables_is_hard(registered=registered):
                def hook_implementation(*args, **kwargs):
                    for impl in registered:
                        impl(*args, **kwargs)

                return hook_implementation

            hook_implementation = closing_over_loop_variables_is_hard()

            hook_implementation.__name__ = hook
            setattr(self, hook, hook_implementation)

        if emission_rate == 'minute':
            def progress():
                return 1.0  # a fake value
        else:
            def progress():
                return self._session_count / self._total_session_count

        # don't compare these strings over and over again!
        self.progress = progress

        self.start_of_simulation(
            emission_rate,
            trading_calendar,
            sessions,
            benchmark_source,
        )

    @property
    def portfolio(self):
        return self._ledger.portfolio

    @property
    def account(self):
        return self._ledger.account

    @property
    def positions(self):
        return self._ledger.position_tracker.positions

    def process_transaction(self, transaction):
        self._ledger.process_transaction(transaction)

    def handle_splits(self, splits):
        self._ledger.process_splits(splits)

    def process_order(self, event):
        self._ledger.process_order(event)

    def process_commission(self, commission):
        self._ledger.process_commission(commission)

    def process_close_position(self, asset, dt, data_portal):
        self._ledger.close_position(self, asset, dt, data_portal)

    def sync_last_sale_prices(self,
                              dt,
                              data_portal,
                              handle_non_market_minutes=False):
        self._ledger.position_tracker.sync_last_sale_prices(
            dt,
            data_portal,
            handle_non_market_minutes=handle_non_market_minutes,
        )

    def handle_minute_close(self, dt, data_portal):
        """
        Handles the close of the given minute in minute emission.

        Parameters
        ----------
        dt : Timestamp
            The minute that is ending

        Returns
        -------
        A minute perf packet.
        """
        self.sync_last_sale_prices(dt, data_portal)

        packet = {}
        self.end_of_bar(
            packet,
            self._ledger,
            dt,
            data_portal,
        )
        return packet

    def handle_market_close(self, dt, data_portal):
        """
        Handles the close of the given day, in both minute and daily emission.
        In daily emission, also updates performance, benchmark and risk metrics
        as it would in handle_minute_close if it were minute emission.

        Parameters
        ----------
        dt : Timestamp
            The minute that is ending

        Returns
        -------
        A daily perf packet.
        """
        completed_session = self._current_session

        if self.emission_rate == 'daily':
            # this method is called for both minutely and daily emissions, but
            # this chunk of code here only applies for daily emissions. (since
            # it's done every minute, elsewhere, for minutely emission).
            self.sync_last_sale_prices(dt, data_portal)

        # increment the day counter before we move markers forward.
        self._session_count += 1

        cal = self._trading_calendar

        # Get the next trading day and, if it is past the bounds of this
        # simulation, return the daily perf packet
        try:
            next_session = cal.next_session_label(
                completed_session
            )
        except NoFurtherDataError:
            next_session = None

        packet = {
            'period_start': self._first_session,
            'period_end': self._last_session,
            'capital_base': self._capital_base,
            'daily_perf': {
                'period_open': self._market_open,
                'period_close': self._market_close,
            },
            'cumulative_perf': {},
            'progress': self.progress,
            'cumulative_risk_metrics': {},
        }
        ledger = self._ledger
        ledger.end_of_session(completed_session)
        self.end_of_session(
            packet,
            ledger,
            completed_session,
            data_portal,
        )

        # If the next trading day is irrelevant, then return the daily packet
        if (next_session is None) or (next_session >= self._last_session):
            return packet

        ledger.process_dividends(
            next_session,
            self._asset_finder,
            self._adjustment_reader,
        )

        self._current_session = next_session
        (self._market_open,
         self._market_close) = cal.open_and_close_for_session(next_session)
        return packet

    def handle_simulation_end(self):
        """
        When the simulation is complete, run the full period risk report
        and send it out on the results socket.
        """
        log.info(
            'Simulated {} trading days\n'
            'first open: {}\n'
            'last close: {}',
            self._session_count,
            self._trading_calendar.session_open(self._first_session),
            self._trading_calendar.session_close(self._last_session),
        )

        packet = {}
        self.end_of_simulation(packet, self._ledger)
        return packet
