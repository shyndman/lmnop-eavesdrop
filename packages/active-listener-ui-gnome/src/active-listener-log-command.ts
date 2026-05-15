const ACTIVE_LISTENER_LOG_TITLE = 'Active Listener logs';
const ACTIVE_LISTENER_LOG_FILTER = String.raw`Active Listener|active-listener\.service|active-listener|eavesdrop@shyndman\.ca|ca\.lmnop\.Eavesdrop`;
const ACTIVE_LISTENER_LOG_JOURNAL_COMMAND = `journalctl --user --since "2 hours ago" --follow --output short-precise | rg --line-buffered -i "${ACTIVE_LISTENER_LOG_FILTER}"`;

export const buildActiveListenerLogCommand = (kittyExecutable: string = 'kitty'): string[] => [
  kittyExecutable,
  '--title',
  ACTIVE_LISTENER_LOG_TITLE,
  'sh',
  '-lc',
  ACTIVE_LISTENER_LOG_JOURNAL_COMMAND,
];
