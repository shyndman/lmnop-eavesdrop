import { FrameRenderer } from './frame-renderer';
import { StateManager } from './state-manager';
import { MessageHandler } from './message-handler';
import { AnimationManager } from './animation-manager';
import { DomManager } from './dom';

// Initialize all renderer components
new FrameRenderer();
const domManager = new DomManager(new AnimationManager());
const uiStateManager = new StateManager(domManager);
const messageHandler = new MessageHandler(uiStateManager);
messageHandler.setupIPC();
await messageHandler.beginProcessingMessages();
