import Gio from 'gi://Gio';
import GLib from 'gi://GLib';

import {
  deriveIndicatorState,
  resolveSetLlmActiveCommandResponse,
  resolveStartOrFinishCommandResponse,
  type IndicatorState,
} from './recording-menu-control.js';
import { resolveActiveListenerServiceErrorSignal } from './active-listener-service-error.js';
import type { TranscriptRun } from './transcript-attributes.js';

const DBUS_BUS_NAME = 'ca.lmnop.Eavesdrop.ActiveListener';
const DBUS_OBJECT_PATH = '/ca/lmnop/Eavesdrop/ActiveListener';
const DBUS_INTERFACE_NAME = 'ca.lmnop.Eavesdrop.ActiveListener1';
const DBUS_STATE_PROPERTY = 'State';
const DBUS_LLM_AVAILABLE_PROPERTY = 'LlmAvailable';
const DBUS_LLM_ACTIVE_PROPERTY = 'LlmActive';
const DBUS_START_OR_FINISH_RECORDING_METHOD = 'StartOrFinishRecording';
const DBUS_SET_LLM_ACTIVE_METHOD = 'SetLlmActive';
const DBUS_TRANSCRIPTION_UPDATED_SIGNAL = 'TranscriptionUpdated';
const DBUS_SPECTRUM_UPDATED_SIGNAL = 'SpectrumUpdated';
const DBUS_PROXY_FLAGS = Gio.DBusProxyFlags.DO_NOT_AUTO_START_AT_CONSTRUCTION;

type DbusTextRun = [string, boolean, boolean];
type DbusTranscriptionUpdatedPayload = [DbusTextRun[]];

export type TranscriptionUpdate = {
  runs: TranscriptRun[];
};

export type ActiveListenerServiceState = {
  indicatorState: IndicatorState;
  servicePresent: boolean;
  phase: string | null;
  llmAvailable: boolean;
  llmActive: boolean;
};

export type ActiveListenerServiceEvents = {
  onStateChanged(state: ActiveListenerServiceState): void;
  onTranscriptionUpdated(update: TranscriptionUpdate): void;
  onSpectrumUpdated(levels: Uint8Array): void;
  onError(title: string, detail: string): void;
};

const normalizeTextRun = ([text, isCommand, isComplete]: DbusTextRun): TranscriptRun => ({
  text,
  isCommand,
  isComplete,
});

export class ActiveListenerServiceClient {
  private readonly events: ActiveListenerServiceEvents;
  private proxy: Gio.DBusProxy | null = null;
  private proxyPropertiesSignalId: number | null = null;
  private proxyDbusSignalId: number | null = null;
  private busWatchId: number | null = null;

  constructor(events: ActiveListenerServiceEvents) {
    this.events = events;
  }

  enable(): void {
    this.busWatchId = Gio.bus_watch_name(
      Gio.BusType.SESSION,
      DBUS_BUS_NAME,
      Gio.BusNameWatcherFlags.NONE,
      () => {
        this.attachProxy();
      },
      () => {
        this.detachProxy();
        this.updateServiceState(false, null, false, false);
      },
    );
  }

  destroy(): void {
    if (this.busWatchId !== null) {
      Gio.bus_unwatch_name(this.busWatchId);
      this.busWatchId = null;
    }

    this.detachProxy();
  }

  runRecordingControlAction(): void {
    const proxy = this.proxy;
    if (proxy === null) {
      return;
    }

    proxy.call(
      DBUS_START_OR_FINISH_RECORDING_METHOD,
      null,
      Gio.DBusCallFlags.NONE,
      -1,
      null,
      (source, result) => {
        const response = resolveStartOrFinishCommandResponse(() => {
          if (source === null || result === null) {
            throw new Error('Active Listener command returned no result');
          }

          const reply = source.call_finish(result);
          return reply.deepUnpack() as [string];
        });

        if (response.kind === 'failure') {
          console.error('Active Listener command failed', response.detail);
          this.events.onError(response.title, response.detail);
          return;
        }

        console.debug(`Active Listener command result ${response.result}`);
      },
    );
  }

  setLlmActive(active: boolean): void {
    const proxy = this.proxy;
    if (proxy === null) {
      return;
    }

    proxy.call(
      DBUS_SET_LLM_ACTIVE_METHOD,
      new GLib.Variant('(b)', [active]),
      Gio.DBusCallFlags.NONE,
      -1,
      null,
      (source, result) => {
        const response = resolveSetLlmActiveCommandResponse(() => {
          if (source === null || result === null) {
            throw new Error('Active Listener command returned no result');
          }

          const reply = source.call_finish(result);
          return reply.deepUnpack() as [boolean];
        });

        if (response.kind === 'failure') {
          console.error('Active Listener command failed', response.detail);
          this.syncIndicatorState();
          this.events.onError(response.title, response.detail);
          return;
        }

        console.debug(`Active Listener command result ${String(response.result)}`);
      },
    );
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
      this.updateServiceState(false, null, false, false);
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
    if (signalName === DBUS_TRANSCRIPTION_UPDATED_SIGNAL) {
      this.handleTranscriptionUpdated(parameters);
      return;
    }

    if (signalName === DBUS_SPECTRUM_UPDATED_SIGNAL) {
      this.handleSpectrumUpdated(parameters);
      return;
    }

    const errorSignal = resolveActiveListenerServiceErrorSignal(
      signalName,
      parameters.deepUnpack(),
    );
    if (errorSignal === null) {
      return;
    }

    console.error(`${errorSignal.title} ${errorSignal.detail}`);
    this.events.onError(errorSignal.title, errorSignal.detail);
  }

  private handleTranscriptionUpdated(parameters: GLib.Variant): void {
    const [runs] = parameters.deepUnpack() as DbusTranscriptionUpdatedPayload;
    this.events.onTranscriptionUpdated({
      runs: runs.map(normalizeTextRun),
    });
  }

  private handleSpectrumUpdated(parameters: GLib.Variant): void {
    const spectrumVariant = parameters.get_child_value(0);
    const byteBuffer = spectrumVariant.get_data_as_bytes();
    this.events.onSpectrumUpdated(Uint8Array.from(byteBuffer.toArray()));
  }

  private syncIndicatorState(): void {
    if (this.proxy === null) {
      this.updateServiceState(false, null, false, false);
      return;
    }

    const phase = this.proxy.get_cached_property(DBUS_STATE_PROPERTY)?.deepUnpack();
    const llmAvailable = this.proxy.get_cached_property(DBUS_LLM_AVAILABLE_PROPERTY)?.deepUnpack();
    const llmActive = this.proxy.get_cached_property(DBUS_LLM_ACTIVE_PROPERTY)?.deepUnpack();
    this.updateServiceState(
      true,
      typeof phase === 'string' ? phase : null,
      llmAvailable === true,
      llmActive === true,
    );
  }

  private updateServiceState(
    servicePresent: boolean,
    phase: string | null,
    llmAvailable: boolean,
    llmActive: boolean,
  ): void {
    this.events.onStateChanged({
      indicatorState: deriveIndicatorState(servicePresent, phase),
      servicePresent,
      phase,
      llmAvailable,
      llmActive,
    });
  }
}
