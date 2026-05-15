import { buildActiveListenerLogCommand } from '../src/active-listener-log-command.ts';

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

const command = buildActiveListenerLogCommand();
const commandText = command.join(' ');
const shellCommand = command.at(-1) ?? '';

assert(command[0] === 'kitty', `expected command to launch kitty, got ${command[0] ?? '<empty>'}`);
assert(command.includes('--title'), 'expected command to set a Kitty title');
assert(command.includes('Active Listener logs'), 'expected command to label the logs window');
assert(command.includes('sh'), 'expected command to invoke a shell');
assert(command.includes('-lc'), 'expected command to run the journal pipeline through sh -lc');
assert(shellCommand.includes('journalctl --user'), 'expected command to read the user journal');
assert(shellCommand.includes('--since "2 hours ago"'), 'expected command to select recent logs');
assert(shellCommand.includes('--follow'), 'expected command to follow logs');
assert(shellCommand.includes('--output short-precise'), 'expected command to use precise journal output');
assert(shellCommand.includes('rg --line-buffered'), 'expected command to filter with line-buffered rg');
assert(!commandText.includes('grep'), 'expected command not to use grep');
assert(shellCommand.includes('active-listener\\.service'), 'expected command to filter active-listener.service');
assert(shellCommand.includes('Active Listener'), 'expected command to filter the extension log prefix');
assert(shellCommand.includes('eavesdrop@shyndman\\.ca'), 'expected command to filter the extension UUID');
assert(shellCommand.includes('ca\\.lmnop\\.Eavesdrop'), 'expected command to filter the D-Bus namespace');

console.log('Active Listener log command contract holds.');
