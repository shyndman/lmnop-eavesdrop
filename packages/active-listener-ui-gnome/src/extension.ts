import GLib from 'gi://GLib';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import { Extension } from 'resource:///org/gnome/shell/extensions/extension.js';

import { buildActiveListenerLogCommand } from './active-listener-log-command.js';
import { ActiveListenerServiceClient } from './active-listener-service-client.js';
import { PanelIndicator } from './panel-indicator.js';
import { callActiveListenerServiceManager, type SystemdMethodName } from './systemd-manager.js';
import { TranscriptOverlayController } from './transcript-overlay.js';

const KITTY_EXECUTABLE_NAME = 'kitty';

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
      restartService: () => {
        void this.runServiceAction('RestartUnit');
      },
      resetUi: () => {
        console.debug('Active Listener indicator reset UI requested');
        this.overlay?.resetUi();
      },
      showLogs: () => {
        this.showLogs();
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

  private showLogs(): void {
    try {
      const kittyExecutable = this.resolveKittyExecutable();
      const [launched] = GLib.spawn_async(
        null,
        buildActiveListenerLogCommand(kittyExecutable),
        null,
        GLib.SpawnFlags.SEARCH_PATH,
        null,
      );
      if (!launched) {
        throw new Error('GLib failed to spawn Kitty');
      }

      console.debug('Active Listener indicator launched logs terminal');
    } catch (error) {
      const detail = error instanceof Error && error.message.length > 0
        ? error.message
        : String(error);
      console.error('Active Listener indicator failed to launch logs terminal', error);
      Main.notifyError('Active Listener logs failed', detail);
    }
  }

  private resolveKittyExecutable(): string {
    const pathExecutable = GLib.find_program_in_path(KITTY_EXECUTABLE_NAME);
    if (pathExecutable !== null) {
      return pathExecutable;
    }

    const homeDirectory = GLib.get_home_dir();
    const candidatePaths = [
      `${homeDirectory}/.local/bin/kitty`,
      `${homeDirectory}/.local/kitty.app/bin/kitty`,
      '/usr/local/bin/kitty',
      '/usr/bin/kitty',
    ];

    for (const candidatePath of candidatePaths) {
      if (GLib.file_test(candidatePath, GLib.FileTest.IS_EXECUTABLE)) {
        return candidatePath;
      }
    }

    return KITTY_EXECUTABLE_NAME;
  }
}
