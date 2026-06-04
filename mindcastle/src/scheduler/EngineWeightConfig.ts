export const DEFAULT_ENGINE_WEIGHTS: Record<string, number> = {
  tech:     1.2,
  health:   1.4,
  finance:  0.8,
  creative: 1.0,
  learning: 1.0,
  writing:  0.9,
  social:   0.7,
  music:    0.8,
};

export function energyThresholdForWeight(weight: number): number {
  return 0.2 + weight * 0.12;
}

export function targetIntervalForWeight(weight: number): number {
  return 48 / weight;
}
