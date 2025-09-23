// Timing constant from ui-state-manager
const WAITING_MESSAGE_DURATION_MS = 2000;

/**
 * Manages the cycling display of waiting messages during command execution.
 *
 * Encapsulates timer lifecycle, message cycling logic, and DOM manipulation
 * for the #command-waiting-messages container.
 */
export class WaitingMessageManager {
  private timerInterval: number | null = null;
  private currentMessageIndex = 0;
  private messages: string[] = [];

  constructor(private containerElement: HTMLElement) {}

  /**
   * Start displaying cycling messages
   */
  start(messages: string[]): void {
    this.stop(); // Clean up any existing timer

    // Use provided messages or default
    this.messages = messages.length > 0 ? [...messages] : ['Generating...'];
    this.currentMessageIndex = 0;

    // Clear existing content and show first message immediately
    this.updateDOM();

    // Start cycling timer
    this.timerInterval = window.setInterval(() => {
      this.cycle();
    }, WAITING_MESSAGE_DURATION_MS);
  }

  /**
   * Stop message cycling and clean up
   */
  stop(): void {
    if (this.timerInterval !== null) {
      clearInterval(this.timerInterval);
      this.timerInterval = null;
    }
    this.messages = [];
    this.currentMessageIndex = 0;
    this.containerElement.innerHTML = '';
  }

  /**
   * Cycle to next message
   */
  private cycle(): void {
    this.currentMessageIndex = (this.currentMessageIndex + 1) % this.messages.length;
    this.updateDOM();
  }

  /**
   * Update DOM to show current message
   */
  private updateDOM(): void {
    // Clear existing content
    this.containerElement.innerHTML = '';

    // Create and add new list item for current message
    const li = document.createElement('li');
    li.textContent = this.messages[this.currentMessageIndex];
    this.containerElement.appendChild(li);
  }

  /**
   * Check if currently running
   */
  isRunning(): boolean {
    return this.timerInterval !== null;
  }
}
