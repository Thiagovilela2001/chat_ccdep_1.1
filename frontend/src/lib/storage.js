export function readStorage(storage, key, fallback = "") {
  try {
    return storage?.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

export function writeStorage(storage, key, value) {
  try {
    storage?.setItem(key, value);
  } catch {
    // A interface continua funcional quando o navegador bloqueia o storage.
  }
}

export function readStoredJson(storage, key, fallback, isValid = () => true) {
  try {
    const rawValue = storage?.getItem(key);
    if (rawValue == null) return fallback;
    const value = JSON.parse(rawValue);
    return isValid(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

