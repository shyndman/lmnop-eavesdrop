import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import St from 'gi://St';

import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';

import {
  deriveLlmRewriteMenuState,
  deriveMenuControlState,
  type IndicatorState,
} from './recording-menu-control.js';
import type { ActiveListenerServiceState } from './active-listener-service-client.js';

const RECORDING_SPIN_DURATION_MS = 2400;
const RECORDING_SPIN_TRANSITION_NAME = 'recording-spin';

export type PanelIndicatorActions = {
  toggleRecording(): void;
  setLlmActive(active: boolean): void;
  openPreferences(): void;
  showOverlayPreview(): void;
  restartService(): void;
  stopService(): void;
};

export class PanelIndicator {
  readonly button: PanelMenu.Button;

  private readonly assetBasePath: string;
  private readonly actions: PanelIndicatorActions;
  private readonly icon: St.Icon;
  private readonly recordingControlItem: PopupMenu.PopupMenuItem;
  private readonly llmRewritingItem: PopupMenu.PopupSwitchMenuItem;
  private readonly restartServiceItem: PopupMenu.PopupMenuItem;
  private readonly stopServiceItem: PopupMenu.PopupMenuItem;
  private indicatorState: IndicatorState = 'absent';
  private servicePresent = false;
  private servicePhase: string | null = null;
  private llmAvailable = false;
  private llmActive = false;
  private applyingLlmToggleState = false;

  constructor(name: string, assetBasePath: string, actions: PanelIndicatorActions) {
    this.assetBasePath = assetBasePath;
    this.actions = actions;
    this.button = new PanelMenu.Button(0.5, name, false);

    this.icon = new St.Icon({
      gicon: this.getStateIcon('absent'),
      style_class: 'system-status-icon',
      accessible_name: 'Active Listener absent',
    });
    this.button.add_child(this.icon);

    this.recordingControlItem = new PopupMenu.PopupMenuItem('No Service');
    this.recordingControlItem.connect('activate', () => {
      this.actions.toggleRecording();
    });

    this.llmRewritingItem = new PopupMenu.PopupSwitchMenuItem('LLM rewriting', false);
    this.llmRewritingItem.connect('toggled', (_item, active) => {
      if (this.applyingLlmToggleState) {
        return;
      }

      this.actions.setLlmActive(active);
    });

    const preferencesItem = new PopupMenu.PopupMenuItem('Preferences');
    preferencesItem.connect('activate', () => {
      this.actions.openPreferences();
    });

    const showOverlayItem = new PopupMenu.PopupMenuItem('Show overlay');
    showOverlayItem.connect('activate', () => {
      this.actions.showOverlayPreview();
    });

    this.restartServiceItem = new PopupMenu.PopupMenuItem('Restart service');
    this.restartServiceItem.connect('activate', () => {
      this.actions.restartService();
    });

    this.stopServiceItem = new PopupMenu.PopupMenuItem('Stop service');
    this.stopServiceItem.connect('activate', () => {
      this.actions.stopService();
    });

    if (!(this.button.menu instanceof PopupMenu.PopupMenu)) {
      throw new Error('Active Listener indicator button menu is unavailable');
    }

    this.button.menu.addMenuItem(this.recordingControlItem);
    this.button.menu.addMenuItem(this.llmRewritingItem);
    this.button.menu.addMenuItem(preferencesItem);
    this.button.menu.addMenuItem(showOverlayItem);
    this.button.menu.addMenuItem(this.restartServiceItem);
    this.button.menu.addMenuItem(this.stopServiceItem);
    this.updateMenuSensitivity();
  }

  destroy(): void {
    this.stopRecordingAnimation();
    this.button.destroy();
  }

  setState(serviceState: ActiveListenerServiceState): void {
    const previousState = this.indicatorState;
    const stateChanged = serviceState.indicatorState !== previousState;

    this.indicatorState = serviceState.indicatorState;
    this.servicePresent = serviceState.servicePresent;
    this.servicePhase = serviceState.phase;
    this.llmAvailable = serviceState.llmAvailable;
    this.llmActive = serviceState.llmActive;
    this.updateMenuSensitivity();

    if (previousState === 'recording' && serviceState.indicatorState !== 'recording') {
      this.stopRecordingAnimation();
    }

    if (stateChanged) {
      this.icon.gicon = this.getStateIcon(serviceState.indicatorState);
      this.icon.accessible_name = `Active Listener ${serviceState.indicatorState}`;
    }

    if (previousState !== 'recording' && serviceState.indicatorState === 'recording') {
      this.startRecordingAnimation();
    }

    if (!stateChanged) {
      return;
    }

    console.debug(`Active Listener indicator state ${serviceState.indicatorState}`);
  }

  private startRecordingAnimation(): void {
    this.icon.remove_transition(RECORDING_SPIN_TRANSITION_NAME);
    this.icon.remove_all_transitions();
    this.icon.set_pivot_point(0.5, 0.5);
    this.icon.rotation_angle_z = 0;

    const transition = Clutter.PropertyTransition.new_for_actor(this.icon, 'rotation-angle-z');
    transition.set_from(0);
    transition.set_to(360);
    transition.set_duration(RECORDING_SPIN_DURATION_MS);
    transition.set_progress_mode(Clutter.AnimationMode.LINEAR);
    transition.set_repeat_count(-1);
    this.icon.add_transition(RECORDING_SPIN_TRANSITION_NAME, transition);

    console.debug('Active Listener indicator started recording animation');
  }

  private stopRecordingAnimation(): void {
    this.icon.remove_transition(RECORDING_SPIN_TRANSITION_NAME);
    this.icon.remove_all_transitions();
    this.icon.rotation_angle_z = 0;

    console.debug('Active Listener indicator stopped recording animation');
  }

  private updateMenuSensitivity(): void {
    const controlState = deriveMenuControlState(this.servicePresent, this.servicePhase);
    const llmRewriteState = deriveLlmRewriteMenuState(
      this.servicePresent,
      this.llmAvailable,
      this.llmActive,
    );
    this.recordingControlItem.label.set_text(controlState.label);
    this.recordingControlItem.sensitive = controlState.enabled;
    this.llmRewritingItem.visible = llmRewriteState.visible;
    this.llmRewritingItem.sensitive = llmRewriteState.enabled;
    this.applyingLlmToggleState = true;
    this.llmRewritingItem.setToggleState(llmRewriteState.active);
    this.applyingLlmToggleState = false;
    this.restartServiceItem.sensitive = true;
    this.stopServiceItem.sensitive = this.servicePresent;
  }

  private getStateIcon(state: IndicatorState): Gio.FileIcon {
    const iconPath = `${this.assetBasePath}/assets/reel-to-reel-${state}.svg`;
    return new Gio.FileIcon({
      file: Gio.File.new_for_path(iconPath),
    });
  }
}
