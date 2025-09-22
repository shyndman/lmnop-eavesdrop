import { Message, MessageType } from '../../messages';
import { UIStateManager } from './ui-state-manager';

export class MessageHandler {
  constructor(private uiStateManager: UIStateManager) {}

  async handleMessage(message: Message): Promise<void> {
    switch (message.type) {
      case MessageType.APPEND_SEGMENTS:
        await this.uiStateManager.appendSegments(
          message.target_mode,
          message.completed_segments,
          message.in_progress_segment
        );
        break;

      case MessageType.CHANGE_MODE:
        this.uiStateManager.changeMode(message.target_mode);
        break;


      case MessageType.SET_STRING:
        await this.uiStateManager.setContent(message.target_mode, message.content);
        break;

      case MessageType.COMMAND_EXECUTING:
        this.uiStateManager.startCommandExecution(message.waiting_messages);
        break;

      case MessageType.COMMIT_OPERATION:
        await this.uiStateManager.commitOperation(message.cancelled);
        break;

      default:
        // TypeScript will error if we miss any enum cases above
        const _exhaustive: never = message;
        window.api.logger.warn('Unknown message type:', _exhaustive);
        break;
    }
  }

  setupIPC(): void {
    // Set up IPC listener for messages from main process
    window.electron.ipcRenderer.on('python-data', async (_event, data) => {
      try {
        await this.handleMessage(data);
      } catch (error) {
        window.api.logger.error('Error handling message:', error, 'Data:', data);
      }
    });
  }
}