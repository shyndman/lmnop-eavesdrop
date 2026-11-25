export const DESKTOP_DBUS_OBJECT_PATH = '/org/gnome/Shell/Extensions/Eavesdrop';
export const DESKTOP_DBUS_INTERFACE_NAME = 'org.gnome.Shell.Extensions.Eavesdrop';
export const DESKTOP_DBUS_ERROR_NAME = `${DESKTOP_DBUS_INTERFACE_NAME}.Error`;

export const DESKTOP_DBUS_INTERFACE_XML = `
<node>
  <interface name="${DESKTOP_DBUS_INTERFACE_NAME}">
    <method name="GetFocusedWindow">
      <arg type="t" direction="out" name="id"/>
      <arg type="s" direction="out" name="wm_class"/>
    </method>
    <method name="GetClipboardText">
      <arg type="s" direction="out" name="text"/>
    </method>
    <method name="SetClipboardText">
      <arg type="s" direction="in" name="text"/>
    </method>
  </interface>
</node>
`.trim();

export type FocusedWindowSource = {
  get_id(): number;
  get_wm_class(): string | null;
};

export type FocusedWindowTuple = [id: number, wmClass: string];

export const resolveFocusedWindowTuple = (
  focusedWindow: FocusedWindowSource | null,
): FocusedWindowTuple | null => {
  if (focusedWindow === null) {
    return null;
  }

  return [focusedWindow.get_id(), focusedWindow.get_wm_class() ?? ''];
};

export const normalizeClipboardText = (text: string | null): string => text ?? '';
