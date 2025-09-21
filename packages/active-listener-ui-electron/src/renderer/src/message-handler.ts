import { Message, MessageType } from '../../messages';

export function handleMessage(message: Message): void {
  switch (message.type) {
    case MessageType.APPEND_SEGMENTS:
      // TODO: Implement segment appending logic
      console.log('append_segments:', message);
      break;

    case MessageType.CHANGE_MODE:
      // TODO: Implement mode switching logic
      console.log('change_mode:', message);
      break;

    case MessageType.SET_SEGMENTS:
      // TODO: Implement segments replacement logic
      console.log('set_segments:', message);
      break;

    case MessageType.SET_STRING:
      // TODO: Implement string content replacement logic
      console.log('set_string:', message);
      break;

    case MessageType.COMMAND_EXECUTED:
      // TODO: Implement command execution visual feedback
      console.log('command_executed:', message);
      break;

    case MessageType.COMMIT_OPERATION:
      // TODO: Implement operation commit and reset logic
      console.log('commit_operation:', message);
      break;

    default:
      // TypeScript will error if we miss any enum cases above
      const _exhaustive: never = message;
      console.warn('Unknown message type:', _exhaustive);
      break;
  }
}

// Set up IPC listener for messages from main process
window.electron.ipcRenderer.on('python-data', (_event, data) => {
  try {
    handleMessage(data);
  } catch (error) {
    console.error('Error handling message:', error, 'Data:', data);
  }
});