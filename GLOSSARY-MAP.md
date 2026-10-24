# Glossary Map

## Contexts

- [Server](packages/server/GLOSSARY.md): recognition, speech detection,
  and segment completion.
- [Client](packages/client/GLOSSARY.md): live transcription, file
  transcription, and subscriptions.
- [Wire](packages/wire/GLOSSARY.md): shared transcript vocabulary.
- [Common](packages/common/GLOSSARY.md): shared utilities and pointers
  to domain vocabulary.
- [Active Listener](packages/active-listener/GLOSSARY.md): workstation
  dictation, text shaping, and rewriting.
- [GNOME UI](packages/active-listener-ui-gnome/GLOSSARY.md): recording
  controls and transcript presentation.

## Relationships

- Server and Client share Wire's transcript vocabulary.
- Server produces transcription results; Client receives them.
- Active Listener uses Client for dictation transcription.
- Active Listener supplies recording state, transcript updates, and
  spectrum levels to GNOME UI; GNOME UI sends recording and rewrite
  controls to Active Listener.
- Common supplies utilities; domain terms remain with their owners.
