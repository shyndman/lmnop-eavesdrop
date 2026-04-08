import { build } from 'esbuild';
import AdmZip from 'adm-zip';
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(__dirname, '.out');
const distDir = resolve(outDir, 'dist');
const assetsDir = resolve(distDir, 'assets');
const metadataPath = resolve(__dirname, 'metadata.json');
const sourceIconPath = resolve(__dirname, '..', '..', 'assets', 'reel-to-reel.svg');
const metadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
const iconTemplate = readFileSync(sourceIconPath, 'utf8');

const STATE_COLORS = {
  absent: '#787878',
  idle: '#8F5FE8',
  recording: '#E14B50',
};

function writeStateIcon(state, color) {
  const icon = iconTemplate.replace('fill="rgb(4,4,4)"', `fill="${color}"`);
  writeFileSync(resolve(assetsDir, `reel-to-reel-${state}.svg`), icon);
}

async function main() {
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(assetsDir, { recursive: true });

  await build({
    entryPoints: [resolve(__dirname, 'src', 'extension.ts')],
    outfile: resolve(distDir, 'extension.js'),
    format: 'esm',
    platform: 'neutral',
    bundle: false,
    target: 'es2023',
    sourcemap: false,
  });

  writeFileSync(resolve(distDir, 'metadata.json'), JSON.stringify(metadata, null, 2) + '\n');

  for (const [state, color] of Object.entries(STATE_COLORS)) {
    writeStateIcon(state, color);
  }

  const zipPath = resolve(outDir, `${metadata.uuid}.zip`);
  const zip = new AdmZip();
  zip.addLocalFolder(distDir);
  zip.writeZip(zipPath);

  console.log(`Built ${zipPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
