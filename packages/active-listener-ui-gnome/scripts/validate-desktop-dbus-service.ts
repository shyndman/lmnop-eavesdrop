import {
  DESKTOP_DBUS_INTERFACE_NAME,
  DESKTOP_DBUS_INTERFACE_XML,
  DESKTOP_DBUS_OBJECT_PATH,
  normalizeClipboardText,
  resolveFocusedWindowTuple,
} from '../src/desktop-dbus-contract.ts';

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

assert(
  DESKTOP_DBUS_OBJECT_PATH === '/org/gnome/Shell/Extensions/Eavesdrop',
  `unexpected object path: ${DESKTOP_DBUS_OBJECT_PATH}`,
);
assert(
  DESKTOP_DBUS_INTERFACE_NAME === 'org.gnome.Shell.Extensions.Eavesdrop',
  `unexpected interface name: ${DESKTOP_DBUS_INTERFACE_NAME}`,
);
assert(
  DESKTOP_DBUS_INTERFACE_XML.includes('name="GetFocusedWindow"'),
  'expected GetFocusedWindow method in interface XML',
);
assert(
  DESKTOP_DBUS_INTERFACE_XML.includes('name="GetClipboardText"'),
  'expected GetClipboardText method in interface XML',
);
assert(
  DESKTOP_DBUS_INTERFACE_XML.includes('name="SetClipboardText"'),
  'expected SetClipboardText method in interface XML',
);

const missingWindow = resolveFocusedWindowTuple(null);
assert(missingWindow === null, 'expected null when no window is focused');

const ternWindow = resolveFocusedWindowTuple({
  get_id: () => 3918122167,
  get_wm_class: () => 'so.stencil.tern',
});
assert(
  ternWindow !== null && ternWindow[0] === 3918122167 && ternWindow[1] === 'so.stencil.tern',
  `expected focused window tuple, got ${JSON.stringify(ternWindow)}`,
);

const classlessWindow = resolveFocusedWindowTuple({
  get_id: () => 42,
  get_wm_class: () => null,
});
assert(
  classlessWindow !== null && classlessWindow[0] === 42 && classlessWindow[1] === '',
  `expected empty wm_class fallback, got ${JSON.stringify(classlessWindow)}`,
);

assert(normalizeClipboardText('hello') === 'hello', 'expected clipboard text passthrough');
assert(normalizeClipboardText(null) === '', 'expected null clipboard text to normalize to empty');

console.log('Active Listener desktop D-Bus contract holds.');
