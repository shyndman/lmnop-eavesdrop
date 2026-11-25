/**
 * GNOME Shell session D-Bus service providing desktop primitives for active-listener.
 *
 * Exports focused-window lookup and on-demand CLIPBOARD snapshot/restore under
 * `org.gnome.Shell` at `/org/gnome/Shell/Extensions/Eavesdrop`, keeping active-listener
 * independent of third-party GNOME extensions.
 */

import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import St from 'gi://St';

const DESKTOP_DBUS_OBJECT_PATH = '/org/gnome/Shell/Extensions/Eavesdrop';
const DESKTOP_DBUS_INTERFACE_NAME = 'org.gnome.Shell.Extensions.Eavesdrop';
const DESKTOP_DBUS_ERROR_NAME = `${DESKTOP_DBUS_INTERFACE_NAME}.Error`;
const NO_FOCUSED_WINDOW_MESSAGE = 'No focused window';

/** Text MIME types restored via `set_text` so both Wayland and XWayland apps receive standard text targets. */
const TEXT_MIME_TYPES = new Set([
  'text/plain;charset=utf-8',
  'text/plain',
  'UTF8_STRING',
  'STRING',
]);

/** Preferred MIME order when snapshotting CLIPBOARD so rich/binary items win over plain-text fallbacks. */
const PREFERRED_CLIPBOARD_MIME_TYPES = [
  'x-special/gnome-copied-files',
  'text/uri-list',
  'image/png',
  'image/jxl',
  'image/webp',
  'image/avif',
  'image/jpeg',
  'text/plain;charset=utf-8',
  'text/plain',
  'UTF8_STRING',
  'STRING',
] as const;

const DESKTOP_DBUS_INTERFACE_XML = `
<node>
  <interface name="${DESKTOP_DBUS_INTERFACE_NAME}">
    <method name="GetFocusedWindow">
      <arg type="t" direction="out" name="id"/>
      <arg type="s" direction="out" name="wm_class"/>
    </method>
    <method name="GetClipboardContent">
      <arg type="s" direction="out" name="mime_type"/>
      <arg type="ay" direction="out" name="data"/>
    </method>
    <method name="SetClipboardContent">
      <arg type="s" direction="in" name="mime_type"/>
      <arg type="ay" direction="in" name="data"/>
    </method>
    <method name="SetClipboardText">
      <arg type="s" direction="in" name="text"/>
    </method>
  </interface>
</node>
`.trim();

/** `(windowId, wmClass)` tuple packed directly into the D-Bus `(ts)` return signature. */
type FocusedWindowTuple = [id: number, wmClass: string];

/** Pick the best advertised MIME type to snapshot, skipping X11 meta-targets like `TARGETS`. */
const selectClipboardMimeType = (mimeTypes: string[]): string | null => {
  for (const preferred of PREFERRED_CLIPBOARD_MIME_TYPES) {
    if (mimeTypes.includes(preferred)) {
      return preferred;
    }
  }

  return mimeTypes.find((mimeType) => mimeType.includes('/')) ?? null;
};

/**
 * Manages export and teardown of the Eavesdrop desktop D-Bus object inside GNOME Shell.
 *
 * Passed directly to `Gio.DBusExportedObject.wrapJSObject`; GJS dispatches
 * `GetClipboardContent` to `GetClipboardContentAsync` so callback-based `St.Clipboard`
 * reads complete without blocking GNOME Shell.
 */
export class DesktopDbusService {
  private readonly textDecoder = new TextDecoder();
  private exportedObject: Gio.DBusExportedObject | null = null;

  /** Export the D-Bus interface on the user's session bus; no-op if already exported. */
  enable(): void {
    if (this.exportedObject !== null) {
      return;
    }

    const exportedObject = Gio.DBusExportedObject.wrapJSObject(
      DESKTOP_DBUS_INTERFACE_XML,
      this,
    );
    exportedObject.export(Gio.DBus.session, DESKTOP_DBUS_OBJECT_PATH);

    this.exportedObject = exportedObject;
    console.debug(`Active Listener exported desktop D-Bus service at ${DESKTOP_DBUS_OBJECT_PATH}`);
  }

  /** Flush pending D-Bus replies and unexport the object when the extension disables. */
  destroy(): void {
    if (this.exportedObject === null) {
      return;
    }

    this.exportedObject.flush();
    this.exportedObject.unexport();
    this.exportedObject = null;

    console.debug(`Active Listener unexported desktop D-Bus service at ${DESKTOP_DBUS_OBJECT_PATH}`);
  }

  /** Return the focused window's ID and WM_CLASS, or raise a D-Bus error if none is focused. */
  GetFocusedWindow(): FocusedWindowTuple {
    const focusedWindow = global.display.get_focus_window();
    if (focusedWindow === null) {
      console.error('Active Listener desktop D-Bus GetFocusedWindow found no focused window');
      throw GLib.Error.new_literal(
        Gio.dbus_error_quark(),
        Gio.DBusError.FAILED,
        NO_FOCUSED_WINDOW_MESSAGE,
      );
    }

    const windowId = focusedWindow.get_id();
    const wmClass = focusedWindow.get_wm_class() ?? '';
    console.debug(
      `Active Listener desktop D-Bus GetFocusedWindow id=${windowId} wm_class=${wmClass}`,
    );

    return [windowId, wmClass];
  }

  /** Snapshot the current CLIPBOARD payload asynchronously as `(mime_type, data)`. */
  GetClipboardContentAsync(_params: [], invocation: Gio.DBusMethodInvocation): void {
    try {
      const clipboard = St.Clipboard.get_default();
      const mimeType = selectClipboardMimeType(clipboard.get_mimetypes(St.ClipboardType.CLIPBOARD));
      if (mimeType === null) {
        console.debug('Active Listener desktop D-Bus GetClipboardContent clipboard is empty');
        invocation.return_value(new GLib.Variant('(say)', ['', new Uint8Array()]));
        return;
      }

      // GNOME Shell passes NULL if the selection owner vanished or cannot serve the requested
      // MIME type (even though @girs/st-17 omits `(nullable)`).
      clipboard.get_content(
        St.ClipboardType.CLIPBOARD,
        mimeType,
        (_clipboard: St.Clipboard, bytes: GLib.Bytes | Uint8Array | null): void => {
          try {
            if (bytes === null) {
              console.debug(
                `Active Listener desktop D-Bus GetClipboardContent returned no bytes for ${mimeType}`,
              );
              invocation.return_value(new GLib.Variant('(say)', ['', new Uint8Array()]));
              return;
            }

            const payload = bytes instanceof Uint8Array ? bytes : bytes.toArray();
            console.debug(
              `Active Listener desktop D-Bus GetClipboardContent mime_type=${mimeType} bytes=${payload.length}`,
            );
            invocation.return_value(new GLib.Variant('(say)', [mimeType, payload]));
          } catch (callbackError) {
            this.returnClipboardReadError(invocation, callbackError);
          }
        },
      );
    } catch (error) {
      this.returnClipboardReadError(invocation, error);
    }
  }

  /** Restore a previously snapshotted CLIPBOARD payload, preserving non-text MIME types. */
  SetClipboardContent(mimeType: string, data: Uint8Array): void {
    const clipboard = St.Clipboard.get_default();
    if (TEXT_MIME_TYPES.has(mimeType)) {
      clipboard.set_text(St.ClipboardType.CLIPBOARD, this.textDecoder.decode(data));
    } else {
      clipboard.set_content(St.ClipboardType.CLIPBOARD, mimeType, data);
    }

    console.debug(
      `Active Listener desktop D-Bus SetClipboardContent mime_type=${mimeType} bytes=${data.length}`,
    );
  }

  /** Stage text in CLIPBOARD only, leaving the PRIMARY middle-click selection untouched. */
  SetClipboardText(text: string): void {
    St.Clipboard.get_default().set_text(St.ClipboardType.CLIPBOARD, text);
    console.debug(`Active Listener desktop D-Bus SetClipboardText length=${text.length}`);
  }

  private returnClipboardReadError(invocation: Gio.DBusMethodInvocation, error: unknown): void {
    const detail = error instanceof Error ? error.message : String(error);
    console.error('Active Listener desktop D-Bus GetClipboardContent failed', error);
    invocation.return_dbus_error(DESKTOP_DBUS_ERROR_NAME, detail);
  }
}
