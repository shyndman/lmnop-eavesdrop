import Clutter from 'gi://Clutter';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import St from 'gi://St';

import { Extension } from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';

const DBUS_BUS_NAME = 'ca.lmnop.Eavesdrop.ActiveListener';
const DBUS_OBJECT_PATH = '/ca/lmnop/Eavesdrop/ActiveListener';
const DBUS_INTERFACE_NAME = 'ca.lmnop.Eavesdrop.ActiveListener1';
const DBUS_STATE_PROPERTY = 'State';
const DBUS_PIPELINE_FAILED_SIGNAL = 'PipelineFailed';
const SYSTEMD_DBUS_BUS_NAME = 'org.freedesktop.systemd1';
const SYSTEMD_DBUS_OBJECT_PATH = '/org/freedesktop/systemd1';
const SYSTEMD_DBUS_INTERFACE_NAME = 'org.freedesktop.systemd1.Manager';
const ACTIVE_LISTENER_SERVICE_NAME = 'active-listener.service';
const SYSTEMD_JOB_MODE = 'replace';
const OVERLAY_MESSAGE = 'Overlay PoC';
const OVERLAY_DISPLAY_DURATION_MS = 2000;
const OVERLAY_BOTTOM_MARGIN_PX = 96;
const OVERLAY_ANIMATION_DURATION_MS = 180;
const RECORDING_SPIN_DURATION_MS = 1200;
const RECORDING_SPIN_TRANSITION_NAME = 'recording-spin';

type ActorEaseOptions = {
  duration: number;
  mode: Clutter.AnimationMode;
  opacity?: number;
  onComplete?: () => void;
};

type IndicatorState = 'absent' | 'idle' | 'recording';
type SystemdMethodName = 'RestartUnit' | 'StopUnit';

const DBUS_PROXY_FLAGS = Gio.DBusProxyFlags.DO_NOT_AUTO_START_AT_CONSTRUCTION;

export default class ActiveListenerIndicatorExtension extends Extension {
  private button: PanelMenu.Button | null = null;
  private icon: St.Icon | null = null;
  private proxy: Gio.DBusProxy | null = null;
  private proxyPropertiesSignalId: number | null = null;
  private proxyDbusSignalId: number | null = null;
  private busWatchId: number | null = null;
  private restartServiceItem: PopupMenu.PopupMenuItem | null = null;
  private stopServiceItem: PopupMenu.PopupMenuItem | null = null;
  private overlay: St.Widget | null = null;
  private overlayLabel: St.Label | null = null;
  private overlayTimeoutId: number | null = null;
  private readonly overlayActor = (actor: St.Widget): St.Widget & {
    ease(options: ActorEaseOptions): void;
    remove_all_transitions(): void;
  } => actor as St.Widget & { ease(options: ActorEaseOptions): void; remove_all_transitions(): void };
  private indicatorState: IndicatorState = 'absent';

  enable(): void {
    this.button = new PanelMenu.Button(0.5, this.metadata.name, false);

    this.icon = new St.Icon({
      gicon: this.getStateIcon('absent'),
      style_class: 'system-status-icon',
      accessible_name: 'Active Listener absent',
    });
    this.button.add_child(this.icon);
    this.addMenuItems();
    this.createOverlay();

    Main.panel.addToStatusArea(this.uuid, this.button);

    this.busWatchId = Gio.bus_watch_name(
      Gio.BusType.SESSION,
      DBUS_BUS_NAME,
      Gio.BusNameWatcherFlags.NONE,
      () => {
        this.attachProxy();
      },
      () => {
        this.detachProxy();
        this.updateIndicator('absent');
      },
    );
  }

  disable(): void {
    if (this.busWatchId !== null) {
      Gio.bus_unwatch_name(this.busWatchId);
      this.busWatchId = null;
    }

    this.detachProxy();
    this.clearOverlayTimeout();
    this.stopRecordingAnimation();

    if (this.overlay !== null) {
      Main.layoutManager.removeChrome(this.overlay);
      this.overlay.destroy();
      this.overlay = null;
    }

    this.overlayLabel = null;
    this.icon = null;
    this.restartServiceItem = null;
    this.stopServiceItem = null;

    this.button?.destroy();
    this.button = null;
  }

  private addMenuItems(): void {
    if (this.button === null) {
      return;
    }

    if (!(this.button.menu instanceof PopupMenu.PopupMenu)) {
      return;
    }

    const preferencesItem = new PopupMenu.PopupMenuItem('Preferences');
    preferencesItem.connect('activate', () => {
      this.openPreferences();
    });

    const showOverlayItem = new PopupMenu.PopupMenuItem('Show overlay');
    showOverlayItem.connect('activate', () => {
      this.showOverlay(OVERLAY_MESSAGE);
    });

    this.restartServiceItem = new PopupMenu.PopupMenuItem('Restart service');
    this.restartServiceItem.connect('activate', () => {
      void this.runServiceAction('RestartUnit');
    });

    this.stopServiceItem = new PopupMenu.PopupMenuItem('Stop service');
    this.stopServiceItem.connect('activate', () => {
      void this.runServiceAction('StopUnit');
    });

    this.button.menu.addMenuItem(preferencesItem);
    this.button.menu.addMenuItem(showOverlayItem);
    this.button.menu.addMenuItem(this.restartServiceItem);
    this.button.menu.addMenuItem(this.stopServiceItem);
    this.updateMenuSensitivity();
  }

  private createOverlay(): void {
    if (this.overlay !== null) {
      return;
    }

    this.overlayLabel = new St.Label({
      text: OVERLAY_MESSAGE,
      x_align: Clutter.ActorAlign.CENTER,
      y_align: Clutter.ActorAlign.CENTER,
      style:
        'color: white;' +
        'font-size: 24px;' +
        'font-weight: 700;' +
        'width: 100%;' +
        'text-align: center;',
    });

    this.overlay = new St.Widget({
      layout_manager: new Clutter.BinLayout(),
      visible: false,
      reactive: false,
      can_focus: false,
      opacity: 0,
      style:
        'background-color: rgba(255, 0, 255, 0.8);' +
        'border-radius: 999px;' +
        'padding: 20px 32px;' +
        'min-width: 420px;' +
        'min-height: 72px;',
    });
    this.overlay.add_child(this.overlayLabel);

    Main.layoutManager.addChrome(this.overlay, { trackFullscreen: true });
    this.overlay.set_position(-10000, -10000);
  }

  private showOverlay(message: string): void {
    if (this.overlay === null || this.overlayLabel === null) {
      return;
    }

    const monitor = Main.layoutManager.primaryMonitor;
    if (monitor === null) {
      return;
    }

    this.clearOverlayTimeout();
    this.overlayLabel.text = message;

    const [, overlayWidth] = this.overlay.get_preferred_width(-1);
    const [, overlayHeight] = this.overlay.get_preferred_height(overlayWidth);
    const x = Math.floor(monitor.x + (monitor.width - overlayWidth) / 2);
    const y = Math.floor(monitor.y + monitor.height - overlayHeight - OVERLAY_BOTTOM_MARGIN_PX);

    const overlayActor = this.overlayActor(this.overlay);
    overlayActor.remove_all_transitions();
    this.overlay.set_position(x, y);
    this.overlay.opacity = 0;
    this.overlay.show();
    overlayActor.ease({
      opacity: 255,
      duration: OVERLAY_ANIMATION_DURATION_MS,
      mode: Clutter.AnimationMode.EASE_OUT_QUAD,
    });

    this.overlayTimeoutId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, OVERLAY_DISPLAY_DURATION_MS, () => {
      this.hideOverlay();
      this.overlayTimeoutId = null;
      return GLib.SOURCE_REMOVE;
    });
  }

  private hideOverlay(): void {
    if (this.overlay === null) {
      return;
    }

    const overlayActor = this.overlayActor(this.overlay);
    overlayActor.remove_all_transitions();
    overlayActor.ease({
      opacity: 0,
      duration: OVERLAY_ANIMATION_DURATION_MS,
      mode: Clutter.AnimationMode.EASE_OUT_QUAD,
      onComplete: () => {
        this.overlay?.hide();
      },
    });
  }

  private clearOverlayTimeout(): void {
    if (this.overlayTimeoutId === null) {
      return;
    }

    GLib.Source.remove(this.overlayTimeoutId);
    this.overlayTimeoutId = null;
  }

  private attachProxy(): void {
    this.detachProxy();

    try {
      this.proxy = Gio.DBusProxy.new_for_bus_sync(
        Gio.BusType.SESSION,
        DBUS_PROXY_FLAGS,
        null,
        DBUS_BUS_NAME,
        DBUS_OBJECT_PATH,
        DBUS_INTERFACE_NAME,
        null,
      );
    } catch (error) {
      console.error('Active Listener indicator failed to create DBus proxy', error);
      this.updateIndicator('absent');
      return;
    }

    this.proxyPropertiesSignalId = this.proxy.connect('g-properties-changed', () => {
      this.syncIndicatorState();
    });
    this.proxyDbusSignalId = this.proxy.connect('g-signal', (_proxy, _senderName, signalName, parameters) => {
      this.handleProxySignal(signalName, parameters);
    });

    console.debug('Active Listener indicator connected to DBus service');
    this.syncIndicatorState();
  }

  private detachProxy(): void {
    if (this.proxy !== null && this.proxyPropertiesSignalId !== null) {
      this.proxy.disconnect(this.proxyPropertiesSignalId);
    }
    if (this.proxy !== null && this.proxyDbusSignalId !== null) {
      this.proxy.disconnect(this.proxyDbusSignalId);
    }

    this.proxyPropertiesSignalId = null;
    this.proxyDbusSignalId = null;
    this.proxy = null;
  }

  private handleProxySignal(signalName: string, parameters: GLib.Variant): void {
    if (signalName !== DBUS_PIPELINE_FAILED_SIGNAL) {
      return;
    }

    const [step, reason] = parameters.deepUnpack() as [string, string];
    const detail = `${step}: ${reason}`;
    console.error(`Active Listener pipeline failed ${detail}`);
    Main.notifyError('Active Listener pipeline failed', detail);
  }

  private syncIndicatorState(): void {
    if (this.proxy === null) {
      this.updateIndicator('absent');
      return;
    }

    const value = this.proxy.get_cached_property(DBUS_STATE_PROPERTY)?.deepUnpack();
    const nextState = value === 'recording' ? 'recording' : 'idle';
    this.updateIndicator(nextState);
  }

  private async runServiceAction(methodName: SystemdMethodName): Promise<void> {
    try {
      await this.callSystemdManager(methodName);
      console.debug(`Active Listener indicator requested ${methodName} for ${ACTIVE_LISTENER_SERVICE_NAME}`);
    } catch (error) {
      console.error(`Active Listener indicator failed to ${methodName} ${ACTIVE_LISTENER_SERVICE_NAME}`, error);
    }
  }

  private callSystemdManager(methodName: SystemdMethodName): Promise<void> {
    return new Promise((resolve, reject) => {
      const connection = Gio.DBus.session;

      connection.call(
        SYSTEMD_DBUS_BUS_NAME,
        SYSTEMD_DBUS_OBJECT_PATH,
        SYSTEMD_DBUS_INTERFACE_NAME,
        methodName,
        new GLib.Variant('(ss)', [ACTIVE_LISTENER_SERVICE_NAME, SYSTEMD_JOB_MODE]),
        null,
        Gio.DBusCallFlags.NONE,
        -1,
        null,
        (source, result) => {
          try {
            source?.call_finish(result);
            resolve();
          } catch (error) {
            reject(error);
          }
        },
      );
    });
  }

  private updateIndicator(state: IndicatorState): void {
    const previousState = this.indicatorState;
    const stateChanged = state !== previousState;
    this.indicatorState = state;
    this.updateMenuSensitivity();

    if (this.icon === null) {
      return;
    }

    if (previousState === 'recording' && state !== 'recording') {
      this.stopRecordingAnimation();
    }

    if (stateChanged) {
      this.icon.gicon = this.getStateIcon(state);
      this.icon.accessible_name = `Active Listener ${state}`;
    }

    if (previousState !== 'recording' && state === 'recording') {
      this.startRecordingAnimation();
    }

    if (!stateChanged) {
      return;
    }

    console.debug(`Active Listener indicator state ${state}`);
  }

  private startRecordingAnimation(): void {
    if (this.icon === null) {
      return;
    }

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
    if (this.icon === null) {
      return;
    }

    this.icon.remove_transition(RECORDING_SPIN_TRANSITION_NAME);
    this.icon.remove_all_transitions();
    this.icon.rotation_angle_z = 0;

    console.debug('Active Listener indicator stopped recording animation');
  }

  private updateMenuSensitivity(): void {
    if (this.restartServiceItem !== null) {
      this.restartServiceItem.sensitive = true;
    }

    if (this.stopServiceItem !== null) {
      this.stopServiceItem.sensitive = this.indicatorState !== 'absent';
    }
  }

  private getStateIcon(state: IndicatorState): Gio.FileIcon {
    const iconPath = `${this.path}/assets/reel-to-reel-${state}.svg`;
    return new Gio.FileIcon({
      file: Gio.File.new_for_path(iconPath),
    });
  }
}
