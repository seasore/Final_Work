import axios from 'axios'

/** 预测/回测等需 PyTorch 推理，CPU 上常超过 30s，避免 axios 误报超时 */
const ML_TIMEOUT_MS = 180000
/** 节假日准确率对多个节假日串行推理，CPU 上可能需数分钟 */
const HOLIDAY_ACCURACY_TIMEOUT_MS = 600000

const api = axios.create({
  baseURL: '/api',
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
})

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token')
  if (token) config.headers.Authorization = `Bearer ${token}`
  return config
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('token')
      localStorage.removeItem('user')
      window.location.href = '/login'
    }
    if (err.code === 'ECONNABORTED' || err.message?.includes?.('timeout')) {
      err.isTimeout = true
    }
    return Promise.reject(err)
  }
)

export const authApi = {
  login: (username, password) => {
    const form = new FormData()
    form.append('username', username)
    form.append('password', password)
    return axios.post('/api/auth/login', form, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      baseURL: '',
    })
  },
  register: (data) => api.post('/auth/register', data),
}

export const predictionApi = {
  /**
   * 返回 data、interval_minutes、n_steps、weather
   * @param steps 默认 96（约一天，快）；192 为完整两天，耗时约翻倍
   */
  getTwoDays: (start, steps = 96) =>
    api.get('/prediction/two-days', { params: { start, steps }, timeout: ML_TIMEOUT_MS }),
}

export const weatherApi = {
  getForecast: (days = 2) =>
    api.get('/weather/forecast', { params: { days }, timeout: ML_TIMEOUT_MS }),
}

export const adminApi = {
  listUsers: (params) => api.get('/admin/users', { params }),
  createUser: (data) => api.post('/admin/users', data),
  updateUserRole: (userId, role) => api.patch(`/admin/users/${userId}/role`, { role }),
  disableUser: (userId, disabled) => api.patch(`/admin/users/${userId}/disable`, { disabled: !!disabled }),
  sendMessage: (data) => api.post('/admin/messages', data),
  unreadReports: (urgency) => api.get('/admin/reports/unread', { params: { urgency } }),
  readReports: (urgency) => api.get('/admin/reports/read', { params: { urgency } }),
  replyReport: (reportId, reply_content) => api.post(`/admin/reports/${reportId}/reply`, { reply_content }),
  logs: (limit) => api.get('/admin/logs', { params: { limit } }),
}

export const messagesApi = {
  list: () => api.get('/messages/'),
  markRead: (msgId) => api.patch(`/messages/${msgId}/read`),
}

export const reportsApi = {
  create: (data) => api.post('/reports/', data),
  myReports: () => api.get('/reports/my'),
}

export const unitsApi = {
  list: () => api.get('/units/'),
  create: (data) => api.post('/units/', data),
  update: (id, data) => api.patch(`/units/${id}`, data),
  delete: (id) => api.delete(`/units/${id}`),
}

export const holidayApi = {
  list: () => api.get('/holiday/'),
  create: (data) => api.post('/holiday/', data),
}

/** 训练集/回测分析（无假数据） */
export const analyticsApi = {
  location: () => api.get('/analytics/location'),
  correlation: (dayType = 'all') => api.get('/analytics/correlation', { params: { day_type: dayType } }),
  /** steps 默认 96（约一天数据量，比 192 快约一半）；可传 48 更快 */
  modelScatter: (start, steps = 96) =>
    api.get('/analytics/model-scatter', { params: { start, steps }, timeout: ML_TIMEOUT_MS }),
  backtest: (start, end) =>
    api.get('/analytics/backtest', { params: { start, end }, timeout: ML_TIMEOUT_MS }),
  heatmap96: (start) => api.get('/analytics/heatmap-96', { params: { start }, timeout: ML_TIMEOUT_MS }),
  holidayLoad: (year) => api.get('/analytics/holiday-load', { params: { year }, timeout: 120000 }),
  holidayAccuracy: (year) =>
    api.get('/analytics/holiday-accuracy', { params: { year }, timeout: HOLIDAY_ACCURACY_TIMEOUT_MS }),
  assessment: (year, month) =>
    api.get('/analytics/assessment', { params: { year, month }, timeout: ML_TIMEOUT_MS }),
}

/** 统一解析 axios/超时错误文案，供各页面 message 使用 */
export function getApiErrorMessage(err) {
  if (err?.isTimeout || err?.code === 'ECONNABORTED' || String(err?.message || '').toLowerCase().includes('timeout')) {
    return '请求超时：模型预测在 CPU 上可能需要数分钟；节假日准确率接口已放宽至约 10 分钟，若仍超时请重启后端后重试。'
  }
  const st = err?.response?.status
  if (st === 502 || st === 504) {
    return '网关/代理超时：后端未启动或预测过久未响应。请确认已在「后端」目录运行 uvicorn（端口 8000），必要时重启前端 dev 并拉长代理超时。'
  }
  if (st === 503) {
    const d = err?.response?.data?.detail
    return typeof d === 'string' ? d : '服务不可用（常见：找不到 best_model.pt 或 scalers.pkl）'
  }
  // Vite 代理连不上后端时，部分环境会表现为 500 且无 detail
  if (st === 500) {
    const d = err?.response?.data?.detail
    if (typeof d !== 'string' || !d.trim()) {
      return '无法访问后端（常见：未启动 API）。请在「后端」目录运行：python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000，并确保 MongoDB 已启动。'
    }
  }
  // 无 HTTP 响应：连接被拒绝、跨域、未走代理等
  if (!err?.response) {
    const code = err?.code
    const msg = String(err?.message || '')
    if (
      code === 'ERR_NETWORK'
      || code === 'ECONNREFUSED'
      || /network/i.test(msg)
      || /failed to fetch/i.test(msg)
    ) {
      return '无法连接后端：请①在「后端」执行 uvicorn（见下方命令）②仅用 npm run dev 打开 http://localhost:5173，不要直接打开 dist 里的 html。③本机需已启动 MongoDB。'
    }
    return msg || '网络异常（无响应），请检查后端是否在 8000 端口运行'
  }
  const d = err?.response?.data?.detail
  if (typeof d === 'string') return d
  if (Array.isArray(d)) return d.map((x) => x.msg || x).join('; ')
  return err?.message || '加载失败'
}

export default api
