import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import { Extension } from 'resource:///org/gnome/shell/extensions/extension.js';

import { ActiveListenerServiceClient } from './active-listener-service-client.js';
import { PanelIndicator } from './panel-indicator.js';
import { callActiveListenerServiceManager, type SystemdMethodName } from './systemd-manager.js';
import { TranscriptOverlayController } from './transcript-overlay.js';

export default class ActiveListenerIndicatorExtension extends Extension {
  private indicator: PanelIndicator | null = null;
  private overlay: TranscriptOverlayController | null = null;
  private serviceClient: ActiveListenerServiceClient | null = null;

  enable(): void {
    this.overlay = new TranscriptOverlayController();
    this.indicator = new PanelIndicator(this.metadata.name, this.path, {
      toggleRecording: () => {
        this.serviceClient?.runRecordingControlAction();
      },
      setLlmActive: (active) => {
        this.serviceClient?.setLlmActive(active);
      },
      openPreferences: () => {
        this.openPreferences();
      },
      showOverlayPreview: () => {
        this.overlay?.showPreview();
      },
      restartService: () => {
        void this.runServiceAction('RestartUnit');
      },
      stopService: () => {
        void this.runServiceAction('StopUnit');
      },
    });
    Main.panel.addToStatusArea(this.uuid, this.indicator.button);

    this.serviceClient = new ActiveListenerServiceClient({
      onStateChanged: (serviceState) => {
        this.indicator?.setState(serviceState);
        this.overlay?.setServiceState(serviceState);
      },
      onTranscriptionUpdated: (update) => {
        this.overlay?.applyTranscriptionUpdate(update);
      },
      onSpectrumUpdated: (levels) => {
        this.overlay?.applySpectrum(levels);
      },
      onError: (title, detail) => {
        Main.notifyError(title, detail);
      },
    });
    this.serviceClient.enable();
  }

  disable(): void {
    this.serviceClient?.destroy();
    this.serviceClient = null;

    this.overlay?.destroy();
    this.overlay = null;

    this.indicator?.destroy();
    this.indicator = null;
  }

  private async runServiceAction(methodName: SystemdMethodName): Promise<void> {
    try {
      await callActiveListenerServiceManager(methodName);
      console.debug(`Active Listener indicator requested ${methodName} for active-listener.service`);
    } catch (error) {
      console.error(`Active Listener indicator failed to ${methodName} active-listener.service`, error);
    }
  }
}
