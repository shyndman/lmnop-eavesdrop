import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import St from 'gi://St';

import {
  DESKTOP_DBUS_ERROR_NAME,
  DESKTOP_DBUS_INTERFACE_XML,
  DESKTOP_DBUS_OBJECT_PATH,
  normalizeClipboardText,
  resolveFocusedWindowTuple,
  type FocusedWindowTuple,
} from './desktop-dbus-contract.js';

const NO_FOCUSED_WINDOW_MESSAGE = 'No focused window';

type DesktopDbusHandler = {
  GetFocusedWindow(): FocusedWindowTuple;
  GetClipboardTextAsync(params: [], invocation: Gio.DBusMethodInvocation): void;
  SetClipboardText(text: string): void;
};

export class DesktopDbusService {
  private exportedObject: Gio.DBusExportedObject | null = null;

  enable(): void {
    if (this.exportedObject !== null) {
      return;
    }

    const handler: DesktopDbusHandler = {
      GetFocusedWindow: () => this.getFocusedWindow(),
      GetClipboardTextAsync: (_params, invocation) => {
        this.getClipboardTextAsync(invocation);
      },
      SetClipboardText: (text) => {
        this.setClipboardText(text);
      },
    };

    const exportedObject = Gio.DBusExportedObject.wrapJSObject(
      DESKTOP_DBUS_INTERFACE_XML,
      handler,
    );
    exportedObject.export(Gio.DBus.session, DESKTOP_DBUS_OBJECT_PATH);

    this.exportedObject = exportedObject;
    console.debug(`Active Listener exported desktop D-Bus service at ${DESKTOP_DBUS_OBJECT_PATH}`);
  }

  destroy(): void {
    if (this.exportedObject === null) {
      return;
    }

    this.exportedObject.flush();
    this.exportedObject.unexport();
    this.exportedObject = null;

    console.debug(`Active Listener unexported desktop D-Bus service at ${DESKTOP_DBUS_OBJECT_PATH}`);
  }

  private getFocusedWindow(): FocusedWindowTuple {
    const focusedWindow = resolveFocusedWindowTuple(global.display.get_focus_window());
    if (focusedWindow === null) {
      console.error('Active Listener desktop D-Bus GetFocusedWindow found no focused window');
      throw GLib.Error.new_literal(
        Gio.dbus_error_quark(),
        Gio.DBusError.FAILED,
        NO_FOCUSED_WINDOW_MESSAGE,
      );
    }

    const [windowId, wmClass] = focusedWindow;
    console.debug(
      `Active Listener desktop D-Bus GetFocusedWindow id=${windowId} wm_class=${wmClass}`,
    );
    return focusedWindow;
  }

  private getClipboardTextAsync(invocation: Gio.DBusMethodInvocation): void {
    try {
      const clipboard = St.Clipboard.get_default();
      clipboard.get_text(St.ClipboardType.CLIPBOARD, (_clipboard, text) => {
        const normalizedText = normalizeClipboardText(text);
        console.debug(
          `Active Listener desktop D-Bus GetClipboardText length=${normalizedText.length}`,
        );
        invocation.return_value(new GLib.Variant('(s)', [normalizedText]));
      });
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      console.error('Active Listener desktop D-Bus GetClipboardText failed', error);
      invocation.return_dbus_error(DESKTOP_DBUS_ERROR_NAME, detail);
    }
  }

  private setClipboardText(text: string): void {
    St.Clipboard.get_default().set_text(St.ClipboardType.CLIPBOARD, text);
    console.debug(`Active Listener desktop D-Bus SetClipboardText length=${text.length}`);
  }
}
