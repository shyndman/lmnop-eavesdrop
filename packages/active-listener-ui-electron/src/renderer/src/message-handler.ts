import { Channel } from 'queueable';
import { Message, MessageType } from '../../messages';
import { StateManager } from './state-manager';

export class MessageHandler {
  private channel: Channel<Message> = new Channel();

  constructor(private uiStateManager: StateManager) {}

  async beginProcessingMessages(): Promise<void> {
    for await (const message of this.channel) {
      switch (message.type) {
        case MessageType.APPEND_SEGMENTS:
          await this.uiStateManager.appendSegments(
            message.target_mode,
            message.completed_segments,
            message.in_progress_segment,
          );
          break;

        case MessageType.CHANGE_MODE:
          this.uiStateManager.changeMode(message.target_mode);
          break;

        case MessageType.SET_STRING:
          await this.uiStateManager.setString(message.target_mode, message.content);
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
  }

  setupIPC(): void {
    // Set up IPC listener for messages from main process
    window.electron.ipcRenderer.on('python-data', async (_event, msg: Message) => {
      try {
        this.channel.push(msg);
      } catch (error) {
        window.api.logger.error('Error handling message:', error, 'Data:', msg);
      }
    });
  }
}
