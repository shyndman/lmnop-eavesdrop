import {
  ACTIVE_LISTENER_AUDIO_ARCHIVE_FAILED_TITLE,
  ACTIVE_LISTENER_PIPELINE_FAILED_TITLE,
  DBUS_AUDIO_ARCHIVE_FAILED_SIGNAL,
  DBUS_PIPELINE_FAILED_SIGNAL,
  resolveActiveListenerServiceErrorSignal,
} from '../src/active-listener-service-error.ts';

const assert = (condition: boolean, message: string): void => {
  if (!condition) {
    throw new Error(message);
  }
};

{
  const errorSignal = resolveActiveListenerServiceErrorSignal(DBUS_PIPELINE_FAILED_SIGNAL, [
    'rewrite_with_llm',
    'timed out',
  ]);

  assert(errorSignal !== null, 'pipeline failure signal should map to a notification');
  assert(errorSignal?.title === ACTIVE_LISTENER_PIPELINE_FAILED_TITLE, 'pipeline failure should use the locked title');
  assert(errorSignal?.detail === 'rewrite_with_llm: timed out', 'pipeline failure should preserve step + reason detail');
  console.log('PASS pipeline failure signal');
}

{
  const errorSignal = resolveActiveListenerServiceErrorSignal(DBUS_AUDIO_ARCHIVE_FAILED_SIGNAL, [
    'encoder exploded',
  ]);

  assert(errorSignal !== null, 'audio archive failure signal should map to a notification');
  assert(errorSignal?.title === ACTIVE_LISTENER_AUDIO_ARCHIVE_FAILED_TITLE, 'audio archive failure should use the locked title');
  assert(errorSignal?.detail === 'encoder exploded', 'audio archive failure should preserve the failure reason');
  console.log('PASS audio archive failure signal');
}

{
  const errorSignal = resolveActiveListenerServiceErrorSignal('TranscriptionUpdated', []);

  assert(errorSignal === null, 'non-error signals should not map to notifications');
  console.log('PASS non-error signal passthrough');
}

console.log('Active Listener service error signal mappings hold.');
