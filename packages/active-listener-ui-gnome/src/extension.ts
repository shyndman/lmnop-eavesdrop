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

    this.restartServiceItem = new PopupMenu.PopupMenuItem('Restart service');
    this.restartServiceItem.connect('activate', () => {
      void this.runServiceAction('RestartUnit');
    });

    this.stopServiceItem = new PopupMenu.PopupMenuItem('Stop service');
    this.stopServiceItem.connect('activate', () => {
      void this.runServiceAction('StopUnit');
    });

    this.button.menu.addMenuItem(preferencesItem);
    this.button.menu.addMenuItem(this.restartServiceItem);
    this.button.menu.addMenuItem(this.stopServiceItem);
    this.updateMenuSensitivity();
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
    const stateChanged = state !== this.indicatorState;
    this.indicatorState = state;
    this.updateMenuSensitivity();

    if (this.icon === null || !stateChanged) {
      return;
    }

    this.icon.gicon = this.getStateIcon(state);
    this.icon.accessible_name = `Active Listener ${state}`;
    console.debug(`Active Listener indicator state ${state}`);
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
