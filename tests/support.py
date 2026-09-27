import pandas as pd

COLUMNS = ["userid", "itemid", "rating", "timestamp"]


def experiment_ratings():
    """100 interactions. A temporal 80/20 cut leaves every user in both sides."""
    rows = []
    timestamp = 0
    for phase in (range(8), range(8, 10)):
        for user in range(10):
            for offset in phase:
                timestamp += 1
                rows.append((f"u{user}", f"i{(user + offset) % 12}", 1.0, timestamp))
    return pd.DataFrame(rows, columns=COLUMNS)
