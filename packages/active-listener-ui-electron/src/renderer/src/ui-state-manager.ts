export class UIStateManager {
  // asrState's bounding box most closely resembles the shape of the window the user sees on screen
  private asrState: HTMLElement;

  constructor() {
    const asrState = document.getElementById('asr-state');
    if (!asrState) {
      throw new Error('asr-state element not found');
    }

    this.asrState = asrState;
    this.setupMouseHover();
  }

  private setupMouseHover(): void {
    // Mouse hover affordance - allows seeing content below window during active states
    this.asrState.addEventListener('mouseenter', () => {
      document.body.classList.add('mouse-is-over');
    });

    this.asrState.addEventListener('mouseleave', () => {
      document.body.classList.remove('mouse-is-over');
    });
  }
}