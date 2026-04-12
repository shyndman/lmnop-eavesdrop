import Adw from 'gi://Adw';
import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import Gtk from 'gi://Gtk';

import { ExtensionPreferences } from 'resource:///org/gnome/Shell/Extensions/js/extensions/prefs.js';

const PROMPT_OVERRIDE_DIRNAME = 'active-listener';
const PROMPT_OVERRIDE_FILENAME = 'system.md';
const FALLBACK_PROMPT_FILENAME = 'rewrite_prompt.md';
const AUTOSAVE_DELAY_MS = 500;

function getPromptOverridePath(): string {
  return GLib.build_filenamev([
    GLib.get_user_config_dir(),
    PROMPT_OVERRIDE_DIRNAME,
    PROMPT_OVERRIDE_FILENAME,
  ]);
}

function getPromptOverrideFile(): Gio.File {
  return Gio.File.new_for_path(getPromptOverridePath());
}

function loadFileContentsUtf8(file: Gio.File): Promise<string> {
  return new Promise((resolve, reject) => {
    file.load_contents_async(null, (_source, result) => {
      try {
        const [, contents] = file.load_contents_finish(result);
        resolve(new TextDecoder('utf-8').decode(contents));
      } catch (error) {
        reject(error);
      }
    });
  });
}

function ensureParentDirectory(file: Gio.File): void {
  const parent = file.get_parent();
  if (parent === null) {
    return;
  }

  if (parent.query_exists(null)) {
    return;
  }

  parent.make_directory_with_parents(null);
}

function writeFileContentsUtf8(file: Gio.File, contents: string): Promise<void> {
  ensureParentDirectory(file);

  const encodedContents = new TextEncoder().encode(contents);
  return new Promise((resolve, reject) => {
    file.replace_contents_bytes_async(
      new GLib.Bytes(encodedContents),
      null,
      false,
      Gio.FileCreateFlags.REPLACE_DESTINATION,
      null,
      (_source, result) => {
        try {
          file.replace_contents_finish(result);
          resolve();
        } catch (error) {
          reject(error);
        }
      },
    );
  });
}

type LoadedPrompt = {
  contents: string;
  source: 'override' | 'fallback';
};

export default class ActiveListenerPreferences extends ExtensionPreferences {
  private readonly promptBuffer = new Gtk.TextBuffer();
  private readonly revertButton = new Gtk.Button({ label: 'Revert' });
  private initialPromptContents = '';
  private persistedPromptContents = '';
  private autosaveSourceId: number | null = null;
  private autosaveInFlight = false;
  private promptRevision = 0;
  private updatingPromptContents = false;

  async fillPreferencesWindow(
    window: Parameters<ExtensionPreferences['fillPreferencesWindow']>[0],
  ): Promise<void> {
    const page = new Adw.PreferencesPage({
      title: 'Settings',
    });
    const rewriteGroup = new Adw.PreferencesGroup({
      title: 'Rewrite',
      description:
        'Active Listener reads the override file first on each rewrite request. If it is absent, prefs seeds this editor from the packaged default prompt.',
    });

    rewriteGroup.add(this.createPromptPathSection());
    rewriteGroup.add(this.createPromptEditor());
    rewriteGroup.add(this.createActionSection());
    page.add(rewriteGroup);
    window.add(page as unknown as Parameters<typeof window.add>[0]);

    this.promptBuffer.connect('changed', () => {
      if (this.updatingPromptContents) {
        return;
      }

      this.promptRevision += 1;
      this.syncActionSensitivity();
      this.scheduleAutosave();
    });

    this.revertButton.connect('clicked', () => {
      void this.revertPromptContents();
    });

    window.connect('close-request', () => {
      this.cancelPendingAutosave();
      void this.flushAutosave();
      return false;
    });

    await this.initializePromptContents().catch(() => undefined);
  }

  private createPromptPathSection(): Gtk.Box {
    const section = new Gtk.Box({
      orientation: Gtk.Orientation.VERTICAL,
      spacing: 6,
      margin_top: 6,
      margin_bottom: 6,
    });
    const title = new Gtk.Label({
      label: 'Prompt override file',
      xalign: 0,
    });
    const pathLabel = new Gtk.Label({
      label: getPromptOverridePath(),
      selectable: true,
      wrap: true,
      xalign: 0,
    });

    section.append(title);
    section.append(pathLabel);
    return section;
  }

  private createPromptEditor(): Gtk.ScrolledWindow {
    const textView = new Gtk.TextView();
    textView.set_buffer(this.promptBuffer);
    textView.set_accepts_tab(false);
    textView.set_monospace(true);
    textView.set_wrap_mode(Gtk.WrapMode.WORD_CHAR);
    textView.set_top_margin(12);
    textView.set_bottom_margin(12);
    textView.set_left_margin(12);
    textView.set_right_margin(12);

    const scrolledWindow = new Gtk.ScrolledWindow({
      hexpand: true,
      margin_top: 6,
      margin_bottom: 6,
    });
    scrolledWindow.set_min_content_height(280);
    scrolledWindow.set_child(textView);

    return scrolledWindow;
  }

  private createActionSection(): Gtk.Box {
    const actionBox = new Gtk.Box({
      orientation: Gtk.Orientation.HORIZONTAL,
      spacing: 12,
      halign: Gtk.Align.END,
      margin_top: 6,
      margin_bottom: 6,
    });

    actionBox.append(this.revertButton);
    return actionBox;
  }

  private getFallbackPromptFile(): Gio.File {
    return Gio.File.new_for_path(GLib.build_filenamev([this.path, 'assets', FALLBACK_PROMPT_FILENAME]));
  }

  private async loadPromptContents(): Promise<LoadedPrompt> {
    const overrideFile = getPromptOverrideFile();
    if (overrideFile.query_exists(null)) {
      return {
        contents: await loadFileContentsUtf8(overrideFile),
        source: 'override',
      };
    }

    return {
      contents: await loadFileContentsUtf8(this.getFallbackPromptFile()),
      source: 'fallback',
    };
  }

  private getCurrentPromptContents(): string {
    const [start, end] = this.promptBuffer.get_bounds();
    return this.promptBuffer.get_text(start, end, true);
  }

  private setPromptContents(contents: string): void {
    this.promptRevision += 1;
    this.updatingPromptContents = true;
    this.promptBuffer.set_text(contents, -1);
    this.updatingPromptContents = false;
  }

  private syncActionSensitivity(): void {
    const hasChanges = this.getCurrentPromptContents() !== this.initialPromptContents;
    this.revertButton.set_sensitive(hasChanges);
  }

  private scheduleAutosave(): void {
    this.cancelPendingAutosave();

    this.autosaveSourceId = GLib.timeout_add(GLib.PRIORITY_DEFAULT, AUTOSAVE_DELAY_MS, () => {
      this.autosaveSourceId = null;
      void this.flushAutosave();
      return GLib.SOURCE_REMOVE;
    });
  }

  private cancelPendingAutosave(): void {
    if (this.autosaveSourceId === null) {
      return;
    }

    GLib.source_remove(this.autosaveSourceId);
    this.autosaveSourceId = null;
  }

  private async initializePromptContents(): Promise<void> {
    this.cancelPendingAutosave();
    this.setActionSensitivity(false);

    try {
      const prompt = await this.loadPromptContents();
      this.initialPromptContents = prompt.contents;
      this.persistedPromptContents = prompt.contents;
      this.setPromptContents(prompt.contents);
      console.info(`Active Listener prefs loaded ${prompt.source} rewrite prompt`);
    } catch (error) {
      console.error('Active Listener prefs failed to load rewrite prompt', error);
      throw error;
    } finally {
      this.syncActionSensitivity();
    }
  }

  private async revertPromptContents(): Promise<void> {
    this.cancelPendingAutosave();
    this.setActionSensitivity(false);
    this.setPromptContents(this.initialPromptContents);
    await this.flushAutosave();
  }

  private async flushAutosave(): Promise<void> {
    if (this.autosaveInFlight) {
      this.scheduleAutosave();
      return;
    }

    const contents = this.getCurrentPromptContents();
    if (contents === this.persistedPromptContents) {
      this.syncActionSensitivity();
      return;
    }

    const revision = this.promptRevision;
    this.autosaveInFlight = true;

    try {
      await writeFileContentsUtf8(getPromptOverrideFile(), contents);
      this.persistedPromptContents = contents;
      if (revision !== this.promptRevision) {
        this.scheduleAutosave();
      }

      console.info(`Active Listener prefs autosaved rewrite prompt to ${getPromptOverridePath()}`);
    } catch (error) {
      console.error('Active Listener prefs failed to autosave rewrite prompt', error);
    } finally {
      this.autosaveInFlight = false;
      this.syncActionSensitivity();
    }
  }

  private setActionSensitivity(sensitive: boolean): void {
    this.revertButton.set_sensitive(sensitive);
  }
}
