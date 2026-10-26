from abc import ABC, abstractmethod
from collections.abc import Iterator

import pandas as pd


class BaseAdapter(ABC):
    """
    Contract between arbitrary source data and the Cortex 2.0
    Canonical Data Model.

    All temporal intervals use half-open semantics:
        [start_ms, end_ms)
    """

    adapter_name: str = "base"
    adapter_version: str = "1.0"

    @abstractmethod
    def available_streams(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def iter_canonical(
        self,
        stream: str,
        *,
        chunksize: int = 1_000_000,
        source_participant_id: str | None = None,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Yield canonical chunks, optionally restricted to one
        source identity and/or temporal interval.
        """
        raise NotImplementedError

    def provenance(self) -> dict:
        return {
            "adapter_name": self.adapter_name,
            "adapter_version": self.adapter_version,
        }
