import Gio from 'gi://Gio';
import GLib from 'gi://GLib';

export type SystemdMethodName = 'RestartUnit' | 'StopUnit';

const SYSTEMD_DBUS_BUS_NAME = 'org.freedesktop.systemd1';
const SYSTEMD_DBUS_OBJECT_PATH = '/org/freedesktop/systemd1';
const SYSTEMD_DBUS_INTERFACE_NAME = 'org.freedesktop.systemd1.Manager';
const ACTIVE_LISTENER_SERVICE_NAME = 'active-listener.service';
const SYSTEMD_JOB_MODE = 'replace';

export const callActiveListenerServiceManager = (methodName: SystemdMethodName): Promise<void> =>
  new Promise((resolve, reject) => {
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
