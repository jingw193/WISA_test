import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import hashlib
import torch
from tqdm.auto import tqdm

from .. import utils
from ..logging import get_logger


logger = get_logger()


def initialize_preprocessor(
    rank: int,
    num_items: int,
    processor_fn: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    save_dir: Optional[str] = None,
    enable_precomputation: bool = False,
    hash_save: bool = False,
    model_name: str = None,
) -> Union["InMemoryDistributedDataPreprocessor", "PrecomputedDistributedDataPreprocessor"]:
    if enable_precomputation:
        return PrecomputedDistributedDataPreprocessor(rank, num_items, processor_fn, save_dir, hash_save, model_name)
    return InMemoryDistributedDataPreprocessor(rank, num_items, processor_fn, save_dir, model_name)


class DistributedDataProcessorMixin:
    def consume(self, *args, **kwargs):
        raise NotImplementedError("DistributedDataProcessorMixin::consume must be implemented by the subclass.")

    def consume_once(self, *args, **kwargs):
        raise NotImplementedError("DistributedDataProcessorMixin::consume_once must be implemented by the subclass.")

    @property
    def requires_data(self):
        raise NotImplementedError("DistributedDataProcessorMixin::requires_data must be implemented by the subclass.")


class InMemoryDistributedDataPreprocessor(DistributedDataProcessorMixin):
    def __init__(
        self, rank: int, num_items: int, processor_fn: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]], save_dir: str = None, model_name: str = None
    ) -> None:
        super().__init__()

        self._rank = rank
        self._num_items = num_items
        self._processor_fn = processor_fn

        self._cached_samples = []
        self._buffer = InMemoryDataBuffer(num_items)
        self._preprocessed_iterator: Union["InMemoryDataIterable", "InMemoryOnceDataIterable"] = None
        if save_dir is not None:
            self._save_dir = pathlib.Path(save_dir)
            self.model_name = model_name

    def consume(
        self,
        data_type: str,
        components: Dict[str, Any],
        data_iterator,
        generator: Optional[torch.Generator] = None,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        if data_type not in self._processor_fn.keys():
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {list(self._processor_fn.keys())}")
        if cache_samples:
            if use_cached_samples:
                raise ValueError("Cannot cache and use cached samples at the same time.")
            if drop_samples:
                raise ValueError("Cannot cache and drop samples at the same time.")

        for i in range(self._num_items):
            if use_cached_samples:
                item = self._cached_samples[i]
            else:
                item = next(data_iterator)
                if cache_samples:
                    self._cached_samples.append(item)
            item = self._processor_fn[data_type](**item, **components, generator=generator)
            self._buffer.add(data_type, item)

        if drop_samples:
            del self._cached_samples
            self._cached_samples = []

        self._preprocessed_iterator = InMemoryDataIterable(self._rank, data_type, self._buffer)
        return iter(self._preprocessed_iterator)

    def consume_hash_dict(
        self,
        data_type: str,
        components: Dict[str, Any],
        data_iterator,
        generator: Optional[torch.Generator] = None,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        if data_type not in self._processor_fn.keys():
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {list(self._processor_fn.keys())}")
        if cache_samples:
            if use_cached_samples:
                raise ValueError("Cannot cache and use cached samples at the same time.")
            if drop_samples:
                raise ValueError("Cannot cache and drop samples at the same time.")

        self._save_dir_data_type = self._save_dir / data_type
        if data_type == "condition" or data_type == "latent":
            self._save_dir_data_type = self._save_dir_data_type / self.model_name

        for i in range(self._num_items):
            if use_cached_samples:
                item = self._cached_samples[i]
            else:
                item = next(data_iterator)
                if cache_samples:
                    self._cached_samples.append(item)

            hash_id = str(hashlib.sha256(item["caption"][0].encode()).hexdigest())
            self._buffer.add(data_type, hash_id)

        if drop_samples:
            del self._cached_samples
            self._cached_samples = []

        self._preprocessed_iterator = PrecomputedHashDataIterable(self._rank, self._save_dir_data_type, data_type, self._buffer)
        return iter(self._preprocessed_iterator)

    def consume_once(
        self,
        data_type: str,
        components: Dict[str, Any],
        data_iterator,
        generator: Optional[torch.Generator] = None,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        if data_type not in self._processor_fn.keys():
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {list(self._processor_fn.keys())}")
        if cache_samples:
            if use_cached_samples:
                raise ValueError("Cannot cache and use cached samples at the same time.")
            if drop_samples:
                raise ValueError("Cannot cache and drop samples at the same time.")

        for i in range(self._num_items):
            if use_cached_samples:
                item = self._cached_samples[i]
            else:
                item = next(data_iterator)
                if cache_samples:
                    self._cached_samples.append(item)
            item = self._processor_fn[data_type](**item, **components, generator=generator)
            self._buffer.add(data_type, item)

        if drop_samples:
            del self._cached_samples
            self._cached_samples = []

        self._preprocessed_iterator = InMemoryOnceDataIterable(self._rank, data_type, self._buffer)
        return iter(self._preprocessed_iterator)

    @property
    def requires_data(self):
        if self._preprocessed_iterator is None:
            return True
        return self._preprocessed_iterator.requires_data


class PrecomputedDistributedDataPreprocessor(DistributedDataProcessorMixin):
    def __init__(
        self,
        rank: int,
        num_items: int,
        processor_fn: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        save_dir: str,
        hash_save: bool,
        model_name: str,
    ) -> None:
        super().__init__()

        self._rank = rank
        self._num_items = num_items
        self._processor_fn = processor_fn
        self._save_dir = pathlib.Path(save_dir)

        self._cached_samples = []
        self._preprocessed_iterator: Union["PrecomputedDataIterable", "PrecomputedOnceDataIterable"] = None

        self._save_dir.mkdir(parents=True, exist_ok=True)
        self.hash_save = hash_save
        if not self.hash_save:
            subdirectories = [f for f in self._save_dir.iterdir() if f.is_dir()]
            utils.delete_files(subdirectories)
        else:
            self._buffer = PrecomputeSaveWithHashDataBuffer(num_items)
            self.model_name = model_name


    def consume(
        self,
        data_type: str,
        components: Dict[str, Any],
        data_iterator,
        generator: Optional[torch.Generator] = None,
        first_samples: bool = False,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        if data_type not in self._processor_fn.keys():
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {list(self._processor_fn.keys())}")
        if cache_samples:
            if use_cached_samples:
                raise ValueError("Cannot cache and use cached samples at the same time.")
            if drop_samples:
                raise ValueError("Cannot cache and drop samples at the same time.")

        for i in tqdm(range(self._num_items), desc=f"Rank {self._rank}", total=self._num_items):
            if use_cached_samples:
                item = self._cached_samples[i]
            else:
                item = next(data_iterator)
                if cache_samples:
                    self._cached_samples.append(item)
            item = self._processor_fn[data_type](**item, **components, generator=generator)
            _save_item(self._rank, i, item, self._save_dir, data_type)

        if drop_samples:
            del self._cached_samples
            self._cached_samples = []

        self._preprocessed_iterator = PrecomputedDataIterable(self._rank, self._save_dir, data_type)
        return iter(self._preprocessed_iterator)

    def consume_save_hash_dict(
        self,
        data_type: str,
        components: Dict[str, Any],
        data_iterator,
        generator: Optional[torch.Generator] = None,
        first_samples: bool = False,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        print(f"rank number : {self._rank}")
        if data_type not in self._processor_fn.keys():
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {list(self._processor_fn.keys())}")
        if cache_samples:
            if use_cached_samples:
                raise ValueError("Cannot cache and use cached samples at the same time.")
            if drop_samples:
                raise ValueError("Cannot cache and drop samples at the same time.")

        self._save_dir_data_type = self._save_dir / data_type
        if data_type == "condition" or data_type == "latent":
            self._save_dir_data_type = self._save_dir_data_type / self.model_name

        self._save_dir_data_type.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(self._num_items), desc=f"Rank {self._rank}", total=self._num_items):
            if use_cached_samples:
                item = self._cached_samples[i]
            else:
                item = next(data_iterator)
                if cache_samples:
                    self._cached_samples.append(item)

            hash_id = str(hashlib.sha256(item["caption"][0].encode()).hexdigest())
            if first_samples:
                if not (self._save_dir_data_type / f"{hash_id}.pt").is_file():
                    item = self._processor_fn[data_type](**item, **components, generator=generator)
                    _save_item_hash_type(hash_id, item, self._save_dir_data_type)

            self._buffer.add(data_type, hash_id)

        if drop_samples:
            del self._cached_samples
            self._cached_samples = []

        self._preprocessed_iterator = PrecomputedHashDataIterable(self._rank, self._save_dir_data_type, data_type, self._buffer)
        return iter(self._preprocessed_iterator)

    def consume_once(
        self,
        data_type: str,
        components: Dict[str, Any],
        data_iterator,
        generator: Optional[torch.Generator] = None,
        first_samples: bool = False,
        cache_samples: bool = False,
        use_cached_samples: bool = False,
        drop_samples: bool = False,
    ) -> Iterable[Dict[str, Any]]:
        if data_type not in self._processor_fn.keys():
            raise ValueError(f"Invalid data type: {data_type}. Supported types: {list(self._processor_fn.keys())}")
        if cache_samples:
            if use_cached_samples:
                raise ValueError("Cannot cache and use cached samples at the same time.")
            if drop_samples:
                raise ValueError("Cannot cache and drop samples at the same time.")

        for i in tqdm(range(self._num_items), desc=f"Processing data on rank {self._rank}", total=self._num_items):
            if use_cached_samples:
                item = self._cached_samples[i]
            else:
                item = next(data_iterator)
                if cache_samples:
                    self._cached_samples.append(item)
            item = self._processor_fn[data_type](**item, **components, generator=generator)
            _save_item(self._rank, i, item, self._save_dir, data_type)

        if drop_samples:
            del self._cached_samples
            self._cached_samples = []

        self._preprocessed_iterator = PrecomputedOnceDataIterable(self._rank, self._save_dir, data_type)
        return iter(self._preprocessed_iterator)

    @property
    def requires_data(self):
        if self._preprocessed_iterator is None:
            return True
        return self._preprocessed_iterator.requires_data


class InMemoryDataIterable:
    """
    An iterator that loads data items from an in-memory buffer. Once all the data is consumed,
    `requires_data` is set to True, indicating that the more data is required and the preprocessor's
    consume method should be called again.
    """

    def __init__(self, rank: int, data_type: str, buffer: "InMemoryDataBuffer") -> None:
        self._rank = rank
        self._data_type = data_type
        self._buffer = buffer

        self._requires_data = False

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        while (length := self._buffer.get_length(self._data_type)) > 0:
            if length <= 1:
                self._requires_data = True
            yield self._buffer.get(self._data_type)

    def __len__(self) -> int:
        return self._buffer.get_length(self._data_type)

    @property
    def requires_data(self):
        return self._requires_data


class InMemoryOnceDataIterable:
    """
    An iterator that loads data items from an in-memory buffer. This iterator will never set
    `requires_data` to True, as it is assumed that all the data was configured to be preprocessed
    by the user. The data will indefinitely be cycled from the buffer.
    """

    def __init__(self, rank: int, data_type: str, buffer: "InMemoryDataBuffer") -> None:
        self._rank = rank
        self._data_type = data_type
        self._buffer = buffer

        self._requires_data = False

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        assert len(self) > 0, "No data available in the buffer."
        while True:
            item = self._buffer.get(self._data_type)
            yield item
            self._buffer.add(self._data_type, item)

    def __len__(self) -> int:
        return self._buffer.get_length(self._data_type)

    @property
    def requires_data(self):
        return self._requires_data


class PrecomputedDataIterable:
    """
    An iterator that loads preconfigured number of data items from disk. Once all the data is
    loaded, `requires_data` is set to True, indicating that the more data is required and
    the preprocessor's consume method should be called again.
    """

    def __init__(self, rank: int, save_dir: str, data_type: str) -> None:
        self._rank = rank
        self._save_dir = pathlib.Path(save_dir)
        self._num_items = len(list(self._save_dir.glob(f"{data_type}-{rank}-*.pt")))
        self._data_type = data_type

        self._requires_data = False

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        for i in range(self._num_items):
            if i == self._num_items - 1:
                self._requires_data = True
            yield _load_item(self._rank, i, self._save_dir, self._data_type)

    def __len__(self) -> int:
        return self._num_items

    @property
    def requires_data(self):
        return self._requires_data


class PrecomputedOnceDataIterable:
    """
    An infinite iterator that loads preprocessed data from disk. Once initialized, this iterator
    will never set `requires_data` to True, as it is assumed that all the data was configured to
    be preprocessed by the user.
    """

    def __init__(self, rank: int, save_dir: str, data_type: str) -> None:
        self._rank = rank
        self._save_dir = pathlib.Path(save_dir)
        self._num_items = len(list(self._save_dir.glob(f"{data_type}-{rank}-*.pt")))
        self._data_type = data_type

        self._requires_data = False

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        index = 0
        while True:
            yield _load_item(self._rank, index, self._save_dir, self._data_type)
            index = (index + 1) % self._num_items

    def __len__(self) -> int:
        return self._num_items

    @property
    def requires_data(self):
        return self._requires_data

class PrecomputedHashDataIterable:
    """
    An infinite iterator that loads preprocessed data from disk. Once initialized, this iterator
    will never set `requires_data` to True, as it is assumed that all the data was configured to
    be preprocessed by the user.
    """

    def __init__(self, rank: int, save_dir: str, data_type: str, buffer: "InMemoryDataBuffer") -> None:
        self._rank = rank
        self._save_dir = pathlib.Path(save_dir)
        self._data_type = data_type
        self._buffer = buffer

        self._requires_data = False

    def __iter__(self) -> Iterable[Dict[str, Any]]:
        while (length := self._buffer.get_length(self._data_type)) > 0:
            if length <= 1:
                self._requires_data = True
            hash_id = self._buffer.get(self._data_type)
            yield _load_item_hash_type(hash_id=hash_id, directory=self._save_dir)

    def __len__(self) -> int:
        return self._buffer.get_length(self._data_type)

    @property
    def requires_data(self):
        return self._requires_data

class InMemoryDataBuffer:
    def __init__(self, max_limit: int = -1) -> None:
        self.max_limit = max_limit
        self.buffer: Dict[str, List[str]] = {}

    def add(self, data_type: str, item: Dict[str, Any]) -> None:
        if data_type not in self.buffer:
            self.buffer[data_type] = []
        if self.max_limit != -1 and len(self.buffer[data_type]) >= self.max_limit:
            logger.log_freq(
                "WARN",
                "IN_MEMORY_DATA_BUFFER_FULL",
                "Buffer is full. Dropping the oldest item. This message will be logged every 64th time this happens.",
                64,
            )
            self.buffer[data_type].pop(0)
        self.buffer[data_type].append(item)

    def get(self, data_type: str) -> Dict[str, Any]:
        return self.buffer[data_type].pop(0)

    def get_length(self, data_type: str) -> int:
        return len(self.buffer[data_type])


class PrecomputeSaveWithHashDataBuffer:
    def __init__(self, max_limit: int = -1) -> None:
        self.max_limit = max_limit
        self.buffer: Dict[str, List[str]] = {}

    def add(self, data_type: str, hash_id: str) -> None:
        if data_type not in self.buffer:
            self.buffer[data_type] = []
        if self.max_limit != -1 and len(self.buffer[data_type]) >= self.max_limit:
            logger.log_freq(
                "WARN",
                "IN_MEMORY_DATA_BUFFER_FULL",
                "Buffer is full. Dropping the oldest item. This message will be logged every 64th time this happens.",
                64,
            )
            self.buffer[data_type].pop(0)
        self.buffer[data_type].append(hash_id)

    def get(self, data_type: str) -> Dict[str, Any]:
        return self.buffer[data_type].pop(0)

    def get_length(self, data_type: str) -> int:
        return len(self.buffer[data_type])

def _save_item(rank: int, index: int, item: Dict[str, Any], directory: pathlib.Path, data_type: str) -> None:
    filename = directory / f"{data_type}-{rank}-{index}.pt"
    torch.save(item, filename.as_posix())

def _save_item_hash_type(hash_id: str, item: Dict[str, Any], directory: pathlib.Path) -> None:
    filename = directory / f"{hash_id}.pt"
    torch.save(item, filename.as_posix())

def _load_item(rank: int, index: int, directory: pathlib.Path, data_type: str) -> Dict[str, Any]:
    filename = directory / f"{data_type}-{rank}-{index}.pt"
    return torch.load(filename.as_posix(), weights_only=True)

def _load_item_hash_type(hash_id: str, directory: pathlib.Path) -> Dict[str, Any]:
    filename = directory / f"{hash_id}.pt"
    return torch.load(filename.as_posix(), weights_only=True)