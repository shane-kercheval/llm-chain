
"""Helper functions and classes."""
import datetime


class Timer:
    """Provides way to time the duration of code within the context manager."""

    def __enter__(self):
        self._start = datetime.datetime.now()
        return self

    def __exit__(self, *args):  # noqa
        self._end = datetime.datetime.now()
        self.interval = self._end - self._start

    def __str__(self):
        return self.formatted(units='seconds', decimal_places=2)

    def formatted(self, units: str = 'seconds', decimal_places: int = 2) -> str:
        """
        Returns a string with the number of seconds that elapsed on the timer. Displays out to
        `decimal_places`.

        Args:
            units:
                format the elapsed time in terms of seconds, minutes, hours
                (currently only supports seconds)
            decimal_places:
                the number of decimal places to display
        """
        if units == 'seconds':
            return f"{self.interval.total_seconds():.{decimal_places}f} seconds"

        raise ValueError("Only suppports seconds.")
