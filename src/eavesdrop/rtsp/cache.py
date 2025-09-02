"""
Thread-safe caching system for RTSP transcription segments.

Provides per-stream caches with dynamic retention based on listener presence.
When streams have no listeners, segments are cached longer to provide meaningful
history. When listeners are present, cache duration is reduced for memory efficiency.
"""

import asyncio
import time
from collections import deque
from typing import TypedDict

from pydantic.dataclasses import dataclass

from ..config import RTSPCacheConfig
from ..logs import get_logger
from ..messages import TranscriptionMessage
from ..transcription.models import Segment


class CacheStats(TypedDict):
  """Statistics for a single stream cache."""

  stream_name: str
  entry_count: int
  oldest_entry_age: float | None
  newest_entry_age: float | None
  cache_duration: float
  has_listeners: bool


class AllCacheStats(TypedDict):
  """Statistics for all stream caches."""

  total_streams: int
  stream_stats: dict[str, CacheStats]


@dataclass
class CacheEntry:
  """Single cached transcription entry with metadata."""

  timestamp: float
  transcription_message: TranscriptionMessage
  segments: list[Segment]
  language: str | None


class StreamCache:
  """
  Thread-safe cache for a single RTSP stream's transcription history.

  Manages automatic expiry based on listener presence and provides
  efficient access to recent transcription segments.
  """

  def __init__(self, stream_name: str, cache_config: RTSPCacheConfig) -> None:
    """
    Initialize stream cache.

    Args:
        stream_name: Name of the stream this cache serves
        cache_config: Cache duration configuration
    """
    self.stream_name = stream_name
    self.cache_config = cache_config
    self._entries: deque[CacheEntry] = deque()
    self._lock = asyncio.Lock()
    self._has_listeners = False
    self.logger = get_logger(f"stream_cache.{stream_name}")

  async def add_transcription(self, segments: list[Segment], language: str | None = None) -> None:
    """
    Add new transcription segments to the cache.

    Args:
        segments: List of transcription segments to cache
        language: Detected or specified language code
    """
    if not segments:
      return

    # Create the message that subscribers will receive
    transcription_message = TranscriptionMessage(
      stream=self.stream_name, segments=segments, language=language
    )

    entry = CacheEntry(
      timestamp=time.time(),
      transcription_message=transcription_message,
      segments=segments,
      language=language,
    )

    async with self._lock:
      self._entries.append(entry)
      await self._cleanup_expired()

    self.logger.debug(
      "Added transcription to cache",
      segment_count=len(segments),
      cache_size=len(self._entries),
      has_listeners=self._has_listeners,
    )

  async def get_recent_messages(
    self, max_age_seconds: float | None = None
  ) -> list[TranscriptionMessage]:
    """
    Get recent transcription messages from the cache.

    Args:
        max_age_seconds: Maximum age of messages to return. If None, uses current cache duration.

    Returns:
        List of transcription messages sorted by timestamp (oldest first)
    """
    if max_age_seconds is None:
      max_age_seconds = self._get_current_cache_duration()

    cutoff_time = time.time() - max_age_seconds

    async with self._lock:
      await self._cleanup_expired()

      recent_entries = [entry for entry in self._entries if entry.timestamp >= cutoff_time]

      self.logger.debug(
        "Retrieved recent messages",
        requested_max_age=max_age_seconds,
        total_entries=len(self._entries),
        recent_entries=len(recent_entries),
      )

      return [entry.transcription_message for entry in recent_entries]

  async def set_listener_presence(self, has_listeners: bool) -> None:
    """
    Update the listener presence status to adjust cache duration.

    Args:
        has_listeners: Whether this stream currently has active listeners
    """
    if self._has_listeners == has_listeners:
      return

    old_status = self._has_listeners
    self._has_listeners = has_listeners

    # Trigger cleanup when switching to shorter cache duration
    if old_status and not has_listeners:
      async with self._lock:
        await self._cleanup_expired()

    duration = self._get_current_cache_duration()
    self.logger.info(
      "Listener presence changed",
      has_listeners=has_listeners,
      cache_duration=duration,
      cache_size=len(self._entries),
    )

  async def clear_cache(self) -> None:
    """Clear all cached entries."""
    async with self._lock:
      entry_count = len(self._entries)
      self._entries.clear()

    self.logger.info("Cache cleared", cleared_entries=entry_count)

  async def get_cache_stats(self) -> CacheStats:
    """Get cache statistics for debugging and monitoring."""
    async with self._lock:
      await self._cleanup_expired()

      if not self._entries:
        return {
          "stream_name": self.stream_name,
          "entry_count": 0,
          "has_listeners": self._has_listeners,
          "cache_duration": self._get_current_cache_duration(),
          "oldest_entry_age": None,
          "newest_entry_age": None,
        }

      current_time = time.time()
      oldest_age = current_time - self._entries[0].timestamp
      newest_age = current_time - self._entries[-1].timestamp

      return {
        "stream_name": self.stream_name,
        "entry_count": len(self._entries),
        "has_listeners": self._has_listeners,
        "cache_duration": self._get_current_cache_duration(),
        "oldest_entry_age": oldest_age,
        "newest_entry_age": newest_age,
      }

  def _get_current_cache_duration(self) -> float:
    """Get the current cache duration based on listener presence."""
    if self._has_listeners:
      return self.cache_config.has_listener_cache_duration
    else:
      return self.cache_config.waiting_for_listener_duration

  async def _cleanup_expired(self) -> None:
    """Remove expired entries from the cache. Must be called with lock held."""
    cache_duration = self._get_current_cache_duration()
    cutoff_time = time.time() - cache_duration

    initial_count = len(self._entries)

    # Remove entries from the left (oldest) until we hit non-expired entries
    while self._entries and self._entries[0].timestamp < cutoff_time:
      self._entries.popleft()

    removed_count = initial_count - len(self._entries)
    if removed_count > 0:
      self.logger.debug(
        "Cleaned up expired cache entries",
        removed_count=removed_count,
        remaining_count=len(self._entries),
        cache_duration=cache_duration,
      )


class RTSPTranscriptionCache:
  """
  Manager for all RTSP stream transcription caches.

  Provides a centralized interface for caching transcription results
  across multiple streams with listener-aware cache management.
  """

  def __init__(self, cache_config: RTSPCacheConfig) -> None:
    """
    Initialize the RTSP transcription cache manager.

    Args:
        cache_config: Cache duration configuration
    """
    self.cache_config = cache_config
    self._stream_caches: dict[str, StreamCache] = {}
    self._lock = asyncio.Lock()
    self.logger = get_logger("rtsp_transcription_cache")

    self.logger.info(
      "Initialized RTSP transcription cache",
      waiting_for_listener_duration=cache_config.waiting_for_listener_duration,
      has_listener_cache_duration=cache_config.has_listener_cache_duration,
    )

  async def add_transcription(
    self, stream_name: str, segments: list[Segment], language: str | None = None
  ) -> None:
    """
    Add transcription segments to the specified stream's cache.

    Args:
        stream_name: Name of the stream
        segments: List of transcription segments to cache
        language: Detected or specified language code
    """
    cache = await self._get_or_create_cache(stream_name)
    await cache.add_transcription(segments, language)

  async def get_recent_messages(
    self, stream_name: str, max_age_seconds: float | None = None
  ) -> list[TranscriptionMessage]:
    """
    Get recent transcription messages for a stream.

    Args:
        stream_name: Name of the stream
        max_age_seconds: Maximum age of messages to return

    Returns:
        List of recent transcription messages
    """
    cache = await self._get_or_create_cache(stream_name)
    return await cache.get_recent_messages(max_age_seconds)

  async def update_listener_presence(self, stream_name: str, has_listeners: bool) -> None:
    """
    Update listener presence for a stream to adjust cache behavior.

    Args:
        stream_name: Name of the stream
        has_listeners: Whether the stream currently has active listeners
    """
    cache = await self._get_or_create_cache(stream_name)
    await cache.set_listener_presence(has_listeners)

  async def clear_stream_cache(self, stream_name: str) -> None:
    """
    Clear the cache for a specific stream.

    Args:
        stream_name: Name of the stream to clear
    """
    async with self._lock:
      if stream_name in self._stream_caches:
        await self._stream_caches[stream_name].clear_cache()

  async def clear_all_caches(self) -> None:
    """Clear all stream caches."""
    async with self._lock:
      for cache in self._stream_caches.values():
        await cache.clear_cache()

    self.logger.info("Cleared all stream caches", stream_count=len(self._stream_caches))

  async def get_all_cache_stats(self) -> AllCacheStats:
    """Get statistics for all stream caches."""
    stats: AllCacheStats = {"total_streams": len(self._stream_caches), "stream_stats": {}}

    async with self._lock:
      for stream_name, cache in self._stream_caches.items():
        stats["stream_stats"][stream_name] = await cache.get_cache_stats()

    return stats

  async def _get_or_create_cache(self, stream_name: str) -> StreamCache:
    """Get or create a cache for the specified stream."""
    async with self._lock:
      if stream_name not in self._stream_caches:
        self._stream_caches[stream_name] = StreamCache(stream_name, self.cache_config)
        self.logger.debug("Created new stream cache", stream_name=stream_name)

      return self._stream_caches[stream_name]
