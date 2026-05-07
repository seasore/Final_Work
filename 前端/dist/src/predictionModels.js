/**
 * 与后端 prediction_service 返回字段一致（训练生成 baseline_checkpoints 后才有值）
 */
export const COMPARE_MODEL_KEYS = [
  { key: 'lstm_mw', short: 'LSTM', color: '#ff7875' },
  { key: 'bilstm_mw', short: 'BiLSTM', color: '#ffc069' },
  { key: 'tcn_mw', short: 'TCN', color: '#95de64' },
  { key: 'bilstm_attention_mw', short: 'BiLSTM-Attn', color: '#b37feb' },
]

/** @param {Record<string, unknown>|undefined} row */
export function activeCompareKeys(row) {
  if (!row) return []
  return COMPARE_MODEL_KEYS.filter(
    ({ key }) => row[key] != null && !Number.isNaN(Number(row[key])),
  )
}
