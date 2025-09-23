import { defineConfig } from 'eslint/config';
import tseslint from '@electron-toolkit/eslint-config-ts';
import eslintConfigPrettier from '@electron-toolkit/eslint-config-prettier';

export default defineConfig(
  { ignores: ['**/node_modules', '**/dist', '**/out'] },
  {},
  tseslint.configs.recommended,
  eslintConfigPrettier,
  {
    rules: {
      'no-case-declarations': 'off',
      'no-multiple-empty-lines': ['error', { max: 1 }],
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          args: 'all',
          argsIgnorePattern: '^_',
          caughtErrors: 'all',
          caughtErrorsIgnorePattern: '^_',
          destructuredArrayIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          ignoreRestSiblings: true,
        },
      ],
    },
  },
);
