/// <reference types="vite/client" />

import type { StateVector } from './store/types';

declare global {
  interface Window {
    __mindcastle?: {
      updateStateVector: (sv: StateVector) => void;
    };
  }
}
