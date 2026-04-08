import Gio from 'gi://Gio';
import St from 'gi://St';

import { Extension } from 'resource:///org/gnome/shell/extensions/extension.js';
import * as Main from 'resource:///org/gnome/shell/ui/main.js';
import * as PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';

const DBUS_BUS_NAME = 'ca.lmnop.Eavesdrop.ActiveListener';
const DBUS_OBJECT_PATH = '/ca/lmnop/Eavesdrop/ActiveListener';
const DBUS_INTERFACE_NAME = 'ca.lmnop.Eavesdrop.ActiveListener1';
const DBUS_STATE_PROPERTY = 'State';

type IndicatorState = 'absent' | 'idle' | 'recording';

const DBUS_PROXY_FLAGS = Gio.DBusProxyFlags.DO_NOT_AUTO_START_AT_CONSTRUCTION;

export default class ActiveListenerIndicatorExtension extends Extension {
  private button: PanelMenu.Button | null = null;
  private icon: St.Icon | null = null;
  private proxy: Gio.DBusProxy | null = null;
  private proxySignalId: number | null = null;
  private busWatchId: number | null = null;
  private indicatorState: IndicatorState = 'absent';

  enable(): void {
    this.button = new PanelMenu.Button(0.0, this.metadata.name, false);
    this.button.reactive = false;
    this.button.can_focus = false;
    this.button.track_hover = false;

    this.icon = new St.Icon({
      gicon: this.getStateIcon('absent'),
      style_class: 'system-status-icon',
      accessible_name: 'Active Listener absent',
    });
    this.button.add_child(this.icon);

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

    this.button?.destroy();
    this.button = null;
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

    this.proxySignalId = this.proxy.connect('g-properties-changed', () => {
      this.syncIndicatorState();
    });

    console.debug('Active Listener indicator connected to DBus service');
    this.syncIndicatorState();
  }

  private detachProxy(): void {
    if (this.proxy !== null && this.proxySignalId !== null) {
      this.proxy.disconnect(this.proxySignalId);
    }

    this.proxySignalId = null;
    this.proxy = null;
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

  private updateIndicator(state: IndicatorState): void {
    if (this.icon === null || state === this.indicatorState) {
      return;
    }

    this.indicatorState = state;
    this.icon.gicon = this.getStateIcon(state);
    this.icon.accessible_name = `Active Listener ${state}`;
    console.debug(`Active Listener indicator state ${state}`);
  }

  private getStateIcon(state: IndicatorState): Gio.FileIcon {
    const iconPath = `${this.path}/assets/reel-to-reel-${state}.svg`;
    return new Gio.FileIcon({
      file: Gio.File.new_for_path(iconPath),
    });
  }
}
