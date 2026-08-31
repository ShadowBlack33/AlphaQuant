export const pct = (v, digits = 2) => (v === null || v === undefined || Number.isNaN(v) ? "--" : `${(v * 100).toFixed(digits)}%`);
export const num = (v, digits = 3) => (v === null || v === undefined || Number.isNaN(v) ? "--" : Number(v).toFixed(digits));
export const badge = (label) => label;
