const second = 1000;
const minute = 60 * second;
const hour = 60 * minute;
const day = 24 * hour;
const TIME_STEPS = [
  ...[1, 2, 5, 10, 20, 50, 100, 200, 500],
  ...[1, 5, 10, 15, 30].map((n) => n * second),
  ...[1, 2, 5, 10, 15, 30].map((n) => n * minute),
  ...[1, 2, 3, 4, 6, 12].map((n) => n * hour),
  ...[1, 2, 3, 7, 14].map((n) => n * day),
];
const numberFormats = new Map();

export function extent(values) {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  return [low, high];
}

function finiteOr(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

function paddedExtent(min, max, padding, singleton, zero = false) {
  const low = zero ? Math.min(0, min) : min;
  const high = zero ? Math.max(0, max) : max;
  const span = high - low;
  const pad = span > 0 ? span * padding : singleton;
  const start = low === 0 && high > 0 ? 0 : finiteOr(low - pad, low);
  const stop = high === 0 && low < 0 ? 0 : finiteOr(high + pad, high);
  return start < stop ? [start, stop] : [low === 0 ? -1 : low / 2, high || 1].sort((a, b) => a - b);
}

function niceStep(raw, integer) {
  if (!(raw > 0) || !Number.isFinite(raw)) return null;
  const power = 10 ** Math.floor(Math.log10(raw));
  const multiplier = [1, 2, 2.5, 5, 10].find((n) => n * power >= raw * (1 - 1e-10));
  const step = (multiplier || 10) * power;
  return integer ? Math.max(1, Math.ceil(step)) : step;
}

function regularTicks(min, max, step) {
  if (!(step > 0) || !Number.isFinite(step)) return min === max ? [min] : [min, max];
  const start = Math.ceil(min / step - 1e-9);
  const end = Math.floor(max / step + 1e-9);
  const length = end - start + 1;
  if (!Number.isFinite(length) || length > 32) return [min, max];
  const values = Array.from({ length: Math.max(0, length) }, (_, i) => (start + i) * step);
  return [...new Set(values)];
}

export function linearScale(min, max, divisions, { padding = 0.06, zero = false, integer = false } = {}) {
  if (zero && min === 0 && max === 0) return { range: [0, 1], values: [0, 0.5, 1], step: 0.5 };
  const singleton = integer ? 0.65 : Math.abs(min) * 0.05 || 1;
  const range = paddedExtent(min, max, padding, singleton, zero);
  const step = niceStep((range[1] - range[0]) / divisions, integer);
  const values = regularTicks(...range, step);
  return { range, values: values.length ? values : [min], step };
}

export function barScale(min, max, divisions, spacing, integer = false) {
  const pad = spacing / 2;
  const range = [finiteOr(min - pad, min), finiteOr(max + pad, max)];
  const step = niceStep((range[1] - range[0]) / divisions, integer);
  const values = regularTicks(min, max, step);
  return { range, values: values.length ? values : [min], step };
}

export function categoryScale(min, max, divisions) {
  const step = Math.max(1, Math.ceil((max - min) / divisions));
  const values = [];
  for (let value = min; value <= max; value += step) values.push(value);
  return { range: [min - 0.5, max + 0.5], values };
}

export function timeScale(min, max, divisions, barSpacing = 0) {
  const pad = barSpacing / 2;
  const range = pad ? [min - pad, max + pad] : paddedExtent(min, max, 0.025, 15 * minute);
  const interval = (range[1] - range[0]) / divisions;
  const step = TIME_STEPS.find((item) => item >= interval);
  if (step) {
    const values = regularTicks(...range, step);
    return { range, values: values.length ? values : [min], step };
  }

  const months = Math.max(1, Math.ceil(interval / (30.4375 * day)));
  const stride = [1, 2, 3, 6, 12].find((item) => item >= months) || niceStep(months / 12, true) * 12;
  const from = new Date(range[0]);
  const year = from.getUTCFullYear();
  const firstMonth = Math.floor((year * 12 + from.getUTCMonth()) / stride) * stride;
  const values = [];
  for (let index = 0; index <= divisions + 2; index++) {
    const date = new Date(0);
    date.setUTCFullYear(0, firstMonth + index * stride, 1);
    const stamp = date.getTime();
    if (stamp > range[1]) break;
    if (stamp >= range[0]) values.push(stamp);
  }
  return { range, values: values.length ? values : [min], step: stride * 30.4375 * day };
}

export function numberLabel(value, step) {
  if (!value) return "0";
  const magnitude = Math.abs(value);
  if (magnitude >= 1e8 || magnitude < 1e-4) {
    const precision = step ? Math.max(1, Math.ceil(Math.log10(magnitude / step))) : 4;
    return value.toExponential(Math.min(16, precision));
  }
  const digits = step ? Math.min(14, Math.max(0, Math.ceil(-Math.log10(step)) + 1)) : 6;
  if (!numberFormats.has(digits)) numberFormats.set(digits, new Intl.NumberFormat("zh-CN", { maximumFractionDigits: digits }));
  return numberFormats.get(digits).format(value);
}

export function timeLabel(value, scale) {
  const iso = new Date(value).toISOString();
  const [start, end] = scale.range.map((stamp) => new Date(stamp).toISOString());
  const step = scale.step;
  if (step >= 365 * day) return iso.slice(0, 4);
  if (step >= 25 * day) return iso.slice(0, 7);
  if (step >= day) return start.slice(0, 4) === end.slice(0, 4) ? iso.slice(5, 10) : iso.slice(0, 10);
  const clock = iso.slice(11, step < second ? 23 : step < minute ? 19 : 16);
  return start.slice(0, 10) === end.slice(0, 10) ? clock : iso.slice(5, 10) + " " + clock;
}

export function position(value, range) {
  const [low, high] = range;
  return Number.isFinite(high - low) ? (value - low) / (high - low)
    : (value / 2 - low / 2) / (high / 2 - low / 2);
}
