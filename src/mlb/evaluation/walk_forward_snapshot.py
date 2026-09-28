from __future__ import annotations

import pandas as pd


class SnapshotStore:
    def __init__(self) -> None:
        self._frame = pd.DataFrame()
        self._keys: set[tuple[object, object]] = set()

    def write(self, frame: pd.DataFrame) -> None:
        pending: list[tuple[object, object]] = []
        for pitcher_id, game_pk in zip(
            frame["pitcher_id"], frame["game_pk"], strict=True
        ):
            key = (pitcher_id, game_pk)
            if key in self._keys or key in pending:
                raise ValueError(
                    f"duplicate walk-forward snapshot for pitcher_id={pitcher_id}, game_pk={game_pk}"
                )
            pending.append(key)

        self._keys.update(pending)
        self._frame = pd.concat([self._frame, frame], ignore_index=True)

    def frame(self) -> pd.DataFrame:
        return self._frame.copy()
