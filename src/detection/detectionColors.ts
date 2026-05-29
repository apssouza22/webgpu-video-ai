const COLORS: ReadonlyArray<[number, number, number]> = [
  [0.23, 0.51, 0.96],
  [0.94, 0.27, 0.27],
  [0.06, 0.73, 0.51],
  [0.96, 0.62, 0.04],
  [0.55, 0.36, 0.96],
  [0.93, 0.28, 0.6],
];

const labelColorMap = new Map<string, [number, number, number]>();
let nextColorIndex = 0;

export function colorForLabel(label: string): [number, number, number] {
  if (!labelColorMap.has(label)) {
    labelColorMap.set(label, COLORS[nextColorIndex % COLORS.length]);
    nextColorIndex++;
  }
  return labelColorMap.get(label)!;
}
