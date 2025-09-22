import { FrameRenderer } from './frame-renderer';
import { UIStateManager } from './ui-state-manager';
import { MessageHandler } from './message-handler';

// Initialize all renderer components
new FrameRenderer();
const uiStateManager = new UIStateManager();
const messageHandler = new MessageHandler(uiStateManager);
messageHandler.setupIPC();